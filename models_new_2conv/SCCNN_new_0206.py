import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from text.symbols import symbols
import models.Constants as Constants
from models_new_2conv.Modules2 import Mish, LinearNorm, ConvNorm, Conv1dGLU, \
                    MultiHeadAttention, StyleAdaptiveLayerNorm, get_sinusoid_encoding_table
from models.VarianceAdaptor import VarianceAdaptor
from models.Loss import StyleSpeechLoss
from utils import get_mask_from_lengths
from torch.nn import functional as F

LOG_FILE = 'SCCNN_0206_debug.txt'
PLOT_DIR = "mel_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_mel_spectrogram(mel, title, filename):
    """
    Mel-spectrogram을 이미지로 저장하는 함수.
    
    Args:
        mel (torch.Tensor): Mel-spectrogram [T, F] 형태 (2D)
        title (str): 그래프 제목
        filename (str): 저장될 파일명
    """
    plt.figure(figsize=(10, 4))
    plt.imshow(mel.cpu().numpy().T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Channels")
    plt.savefig(filename)
    plt.close()

def log_debug(message, log_file=LOG_FILE):
    with open(log_file, "a") as f:
        f.write(message + "\n")
    
########################################
# Low-band 모델
# 저주파 전용 모델: SCCNN_Low
#  - 입력 Mel 은 이미 [:, :, :40] 부분만 들어옴
#  - Decoder 최종 출력도 40채널
########################################

########################################
# 수정된 LowDecoder: 고주파 모델의 구조와 동일하게 convolution/FFT block 적용
########################################
class LowDecoder(nn.Module):
    """
    저주파 모델의 Decoder:
      - 고주파 모델과 동일한 FFT block 기반 구조를 적용하여 [B,T,40] 예측
    """
    def __init__(self, config):
        super(LowDecoder, self).__init__()
        self.max_seq_len = config.max_seq_len
        self.n_layers = config.decoder_layer
        self.d_model = config.decoder_hidden
        self.n_head = config.decoder_head
        self.d_k = config.decoder_hidden // config.decoder_head
        self.d_v = config.decoder_hidden // config.decoder_head
        self.d_inner = config.fft_conv1d_filter_size
        self.fft_kernel = config.fft_conv1d_kernel_size
        self.style_dim = config.style_vector_dim
        self.dropout = config.dropout

        # Prenet: Linear layers + ReLU (또는 Mish) 적용
        self.prenet = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),  
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.d_model)
        )
        # FFT 블록들을 쌓아 Decoder 구성 (HighDecoder와 동일한 구조)
        self.layer_stack = nn.ModuleList([
            FFTBlock(
                d_model=self.d_model,
                d_inner=self.d_inner,
                n_head=self.n_head,
                d_k=self.d_k,
                d_v=self.d_v,
                fft_conv1d_kernel_size=self.fft_kernel,
                style_dim=self.style_dim,
                dropout=self.dropout
            )
            for _ in range(self.n_layers)
        ])
        # 최종 출력: 40 채널 (저주파 영역)
        self.fc_lo = nn.Linear(self.d_model, 40)
    
    def forward(self, enc_seq, style_vector, mask=None):
        # Prenet 통과
        x = self.prenet(enc_seq)
        # mel_mask가 있다면 attention mask 생성
        slf_attn_mask = None
        if mask is not None:
            B, T, _ = x.shape
            slf_attn_mask = mask.unsqueeze(1).expand(-1, T, -1)
        # FFT 블록들을 순차적으로 통과
        for layer in self.layer_stack:
            x, _ = layer(x, style_vector, mask=mask, slf_attn_mask=slf_attn_mask)
        # 최종 선형 계층 적용하여 40채널 출력
        
        mel_low = self.fc_lo(x)
        if mask is not None:
            mel_low = mel_low.masked_fill(mask.unsqueeze(-1), 0.0)
        return mel_low, None

class SCCNN_Low(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.style_encoder = MelStyleEncoder(config)
        self.encoder = Encoder(config)
        self.variance_adaptor = VarianceAdaptor(config)
        self.decoder = LowDecoder(config)  # 최종 출력이 40ch
    
    def parse_batch(self, batch, batch_index=0):
        # 데이터셋에서 이미 mel_target이 [B, T, 40] 형태로 들어오도록 준비
        sid = torch.from_numpy(batch["sid"]).long().cuda()
        text = torch.from_numpy(batch["text"]).long().cuda()
        #Mel-spectrogram에서 처음 40개의 채널만(저주파)
        mel_target_lo = torch.from_numpy(batch["mel_target_lo"]).float().cuda()[:, :, :40]  ## shape [B,T,40]
        
        # Mel-spectrogram 시각화 (저주파)
        plot_filename = os.path.join(PLOT_DIR, f"mel_target_lo_{batch_index}.png")
        plot_mel_spectrogram(mel_target_lo[0], "Low-band Mel-Spectrogram", plot_filename)  # 첫 번째 배치만 저장

        D = torch.from_numpy(batch["D"]).long().cuda()
        log_D = torch.from_numpy(batch["log_D"]).float().cuda()
        f0 = torch.from_numpy(batch["f0"]).float().cuda()
        energy = torch.from_numpy(batch["energy"]).float().cuda()
        src_len = torch.from_numpy(batch["src_len"]).long().cuda()
        mel_len = torch.from_numpy(batch["mel_len"]).long().cuda()
        max_src_len = np.max(batch["src_len"]).astype(np.int32)
        max_mel_len = np.max(batch["mel_len"]).astype(np.int32)
        return sid, text, mel_target_lo, D, log_D, f0, energy, src_len, mel_len, max_src_len, max_mel_len
    

    def forward(self, src_seq, src_len, mel_target_lo, mel_len=None, 
                    d_target=None, p_target=None, e_target=None, max_src_len=None, max_mel_len=None):
        """
        Lo 모델은 mel_target[..., :40]만 다룬다고 가정.
        """
        # mask
        src_mask = get_mask_from_lengths(src_len, max_src_len)
        mel_mask = get_mask_from_lengths(mel_len, max_mel_len) if mel_len is not None else None
        log_debug(f"[SCCNN_Low] Forward src_seq shape: {src_seq.shape}") #[64, 184]
        
        # 1) style vector (in_dim=40인 MelStyleEncoder)
        style_vector = self.style_encoder(mel_target_lo, mel_mask)
        log_debug(f"[SCCNN_Low] Style vector shape: {style_vector.shape}") #[64, 128]
        
        # 2) phoneme encoder
        encoder_output, src_embedded, _ = self.encoder(src_seq, style_vector, src_mask)
        log_debug(f"[SCCNN_Low] Encoder output shape: {encoder_output.shape}")
        
        # # 3) variance adaptor
        acoustic_adaptor_output, d_prediction, p_prediction, e_prediction, mel_len, mel_mask = self.variance_adaptor(
                encoder_output, src_mask, mel_len, mel_mask, 
                        d_target, p_target, e_target, max_mel_len)
        log_debug(f"[SCCNN_Low] Variance adaptor output shape: {acoustic_adaptor_output.shape}")

        # 4) decoder => 최종 (B,T,40)
        mel_prediction, _ = self.decoder(acoustic_adaptor_output, style_vector, mel_mask)
        log_debug(f"[SCCNN_Low] Mel prediction shape: {mel_prediction.shape}") #[64, 936, 40]
        
        return mel_prediction, src_embedded, style_vector, d_prediction, p_prediction, e_prediction, src_mask, mel_mask, mel_len

        

    def inference(self, style_vector, src_seq, src_len=None, max_src_len=None, return_attn=False, return_kernel=False):
        src_mask = get_mask_from_lengths(src_len, max_src_len)
        
        # Encoder
        if return_kernel:
            kernel_params = self.encoder(src_seq, style_vector, src_mask, True)
            return kernel_params
        encoder_output, src_embedded, enc_slf_attn = self.encoder(src_seq, style_vector, src_mask)

        # Variance Adaptor
        acoustic_adaptor_output, d_pred, p_pred, e_pred, \
                mel_len, mel_mask = self.variance_adaptor(encoder_output, src_mask)


        # Decoder => [B,T,40]
        mel_lo, dec_slf_attn = self.decoder(acoustic_adaptor_output, style_vector, mel_mask)

        if return_attn:
            return enc_slf_attn, dec_slf_attn
        return mel_lo, src_embedded, d_pred, p_pred, e_pred, src_mask, mel_mask, mel_len

    def get_style_vector(self, mel_target_lo, mel_len=None):
        mel_mask = get_mask_from_lengths(mel_len) if mel_len is not None else None
        style_vector = self.style_encoder(mel_target_lo, mel_mask)
        return style_vector

    def get_criterion(self):
        return StyleSpeechLoss()


########################################
# High-band 모델
#  - 입력 Mel 은 이미 [:, :, 40:] 부분만 들어옴
#  - Decoder 최종 출력도 40채널
########################################
class SCCNN_High(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.style_encoder = MelStyleEncoder(config)
        self.encoder = Encoder(config)
        self.variance_adaptor = VarianceAdaptor(config)
        self.decoder = HighDecoder(config)  # 최종 출력이 40ch
    
    def parse_batch(self, batch, batch_index=0):
        # 데이터셋에서 이미 mel_target이 [B, T, 40] 형태로 들어오도록 준비 (고주파 부분)
        sid = torch.from_numpy(batch["sid"]).long().cuda()
        text = torch.from_numpy(batch["text"]).long().cuda()
        mel_target_hi = torch.from_numpy(batch["mel_target_hi"]).float().cuda()[:, :, 40:]
        
        # Mel-spectrogram 시각화 (고주파)
        plot_filename = os.path.join(PLOT_DIR, f"mel_target_hi_{batch_index}.png")
        plot_mel_spectrogram(mel_target_hi[0], "High-band Mel-Spectrogram", plot_filename)  # 첫 번째 배치만 저장

        D = torch.from_numpy(batch["D"]).long().cuda()
        log_D = torch.from_numpy(batch["log_D"]).float().cuda()
        f0 = torch.from_numpy(batch["f0"]).float().cuda()
        energy = torch.from_numpy(batch["energy"]).float().cuda()
        src_len = torch.from_numpy(batch["src_len"]).long().cuda()
        mel_len = torch.from_numpy(batch["mel_len"]).long().cuda()
        max_src_len = np.max(batch["src_len"]).astype(np.int32)
        max_mel_len = np.max(batch["mel_len"]).astype(np.int32)
        return sid, text, mel_target_hi, D, log_D, f0, energy, src_len, mel_len, max_src_len, max_mel_len
    
    
    def forward(self, src_seq, src_len, mel_target_hi, mel_len=None, 
                    d_target=None, p_target=None, e_target=None, max_src_len=None, max_mel_len=None):
        """
        Hi 모델은 mel_target[..., 40:]만 다룬다고 가정.
        """
        #mask
        src_mask = get_mask_from_lengths(src_len, max_src_len)
        mel_mask = get_mask_from_lengths(mel_len, max_mel_len) if mel_len is not None else None
        log_debug(f"[SCCNN_High] Forward src_seq shape: {src_seq.shape}")
        
        # 1) style vector
        style_vector = self.style_encoder(mel_target_hi, mel_mask)
        log_debug(f"[SCCNN_High] Style vector shape: {style_vector.shape}")
        
        # 2) phoneme encoder
        encoder_output, src_embedded, _ = self.encoder(src_seq, style_vector, src_mask)
        log_debug(f"[SCCNN_High] Encoder output shape: {encoder_output.shape}")
        
        # 3) variance adaptor
        acoustic_adaptor_output, d_prediction, p_prediction, e_prediction, mel_len, mel_mask = self.variance_adaptor(
                encoder_output, src_mask, mel_len, mel_mask, 
                        d_target, p_target, e_target, max_mel_len)
        log_debug(f"[SCCNN_High] Variance adaptor output shape: {acoustic_adaptor_output.shape}")

        # 4) decoder => 최종 (B,T,40)
        mel_prediction, _ = self.decoder(acoustic_adaptor_output, style_vector, mel_mask)
        log_debug(f"[SCCNN_High] Mel prediction shape: {mel_prediction.shape}")
        
        return mel_prediction, src_embedded, style_vector, d_prediction, p_prediction, e_prediction, src_mask, mel_mask, mel_len

        
    def inference(self, style_vector, src_seq, src_len=None, max_src_len=None, return_attn=False, return_kernel=False):
        src_mask = get_mask_from_lengths(src_len, max_src_len)
        
        # Encoder
        if return_kernel:
            kernel_params = self.encoder(src_seq, style_vector, src_mask, True)
            return kernel_params
        encoder_output, src_embedded, enc_slf_attn = self.encoder(src_seq, style_vector, src_mask)


        # Variance Adaptor
        acoustic_adaptor_output, d_prediction, p_prediction, e_prediction, \
                mel_len, mel_mask = self.variance_adaptor(encoder_output, src_mask)

        # Decoder
        mel_hi, dec_slf_attn = self.decoder(acoustic_adaptor_output, style_vector, mel_mask)

        if return_attn:
            return enc_slf_attn, dec_slf_attn

        return mel_hi, src_embedded, d_prediction, p_prediction, e_prediction, src_mask, mel_mask, mel_len

    def get_style_vector(self, mel_target_hi, mel_len=None):
        mel_mask = get_mask_from_lengths(mel_len) if mel_len is not None else None
        style_vector = self.style_encoder(mel_target_hi, mel_mask)
        return style_vector

    def get_criterion(self):
        return StyleSpeechLoss()
    
    


class Encoder(nn.Module):
    ''' Encoder '''
    def __init__(self, config, n_src_vocab=len(symbols)+1):
        super(Encoder, self).__init__()
        self.max_seq_len = config.max_seq_len
        self.n_layers = config.encoder_layer
        self.d_model = config.encoder_hidden
        self.n_head = config.encoder_head
        self.d_k = config.encoder_hidden // config.encoder_head
        self.d_v = config.encoder_hidden // config.encoder_head
        self.d_hid1 = config.enc_ffn_in_ch_size
        self.d_hid2 = config.enc_ffn_out_ch_size
        self.fft_conv1d_kernel_size = config.fft_conv1d_kernel_size
        self.d_out = config.decoder_hidden
        self.style_dim = config.style_vector_dim
        self.dropout = config.dropout
        self.highlow = config.highlow

        self.src_word_emb = nn.Embedding(n_src_vocab, self.d_model, padding_idx=Constants.PAD)
        self.prenet = Prenet(self.d_model, self.d_model, self.dropout)

        n_position = self.max_seq_len + 1
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, self.d_model).unsqueeze(0), requires_grad = False)

        self.layer_stack = nn.ModuleList([SCFFTBlock(
            self.d_model, self.d_hid1, self.d_hid2, self.n_head, self.d_k, self.d_v, 
            self.style_dim, self.dropout, self.highlow) for _ in range(self.n_layers)])

        self.fc_out = nn.Linear(self.d_model, self.d_out)
        
        self.kernel_predictor = KernelPredictor(self.style_dim, self.n_layers, config.enc_ffn_style_conv1d_kernel_size, 
                                                config.enc_ffn_in_ch_size, config.enc_ffn_out_ch_size)

    def forward(self, src_seq, style_vector, mask, return_kernels=False):
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]
        log_debug(f"[Encoder] text_input shape: {src_seq.shape}") #[64, 184]
        
        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
        log_debug(f"[Encoder] Self-attention mask shape: {slf_attn_mask.shape}") #[64, 184, 184]

        # -- Forward
        # word embedding
        src_embedded = self.src_word_emb(src_seq)
        log_debug(f"[Encoder] Source embedding shape: {src_embedded.shape}") #[64, 184, 256]
        
        # prenet
        src_seq = self.prenet(src_embedded, mask)
        log_debug(f"[Encoder] After Prenet shape: {src_seq.shape}") #[64, 184, 256]
        
        # position encoding
        if src_seq.shape[1] > self.max_seq_len:
            position_embedded = get_sinusoid_encoding_table(src_seq.shape[1], self.d_model)[:src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(src_seq.device)
        else:
            position_embedded = self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
        enc_output = src_seq + position_embedded
        log_debug(f"[Encoder] After adding position encoding shape: {enc_output.shape}") #[64, 184, 256]
    
        
        # fft blocks
        slf_attn = []
        
        kernel_params = self.kernel_predictor(style_vector)
        d_ws, d_gs, d_bs, p_ws, p_gs, p_bs = kernel_params
        log_debug(f"[Encoder] Kernel parameters shapes: {d_ws.shape}, {d_gs.shape}, {d_bs.shape}")
                                            #torch.Size([64, 4, 8, 1, 7]), torch.Size([64, 4, 8]), torch.Size([64, 4, 8])
        
        if return_kernels:
            return kernel_params
        
        for i, enc_layer in enumerate(self.layer_stack):
            log_debug(f"[Encoder] Passing through FFT layer {i + 1}...")
            d_w, d_g, d_b = d_ws[:, i, :, :, :], d_gs[:, i, :], d_bs[:, i, :]
            p_w, p_g, p_b = p_ws[:, i, :, :, :], p_gs[:, i, :], p_bs[:, i, :]
            
            enc_output, enc_slf_attn = enc_layer(
                enc_output, d_w, d_g, d_b, p_w, p_g, p_b,
                mask=mask, 
                slf_attn_mask=slf_attn_mask)
            slf_attn.append(enc_slf_attn)
        log_debug(f"[Encoder] Output shape after FFT layer {i + 1}: {enc_output.shape}")

        # last fc
        enc_output = self.fc_out(enc_output)
        log_debug(f"[Encoder] Final encoder output shape: {enc_output.shape}")
        return enc_output, src_embedded, slf_attn
        
class HighDecoder(nn.Module):
    """
    고주파만 예측하는 Decoder:
      - 기존 SC-CNN or FFTBlock 등을 적용
      - 최종 출력 40채널
    """
    def __init__(self, config):
        super(HighDecoder, self).__init__()
        self.max_seq_len = config.max_seq_len
        self.n_layers = config.decoder_layer
        self.d_model = config.decoder_hidden
        self.n_head = config.decoder_head
        self.d_k = config.decoder_hidden // config.decoder_head
        self.d_v = config.decoder_hidden // config.decoder_head
        self.d_inner = config.fft_conv1d_filter_size
        self.fft_kernel = config.fft_conv1d_kernel_size
        self.style_dim = config.style_vector_dim
        self.dropout = config.dropout

        # prenet
        self.prenet = nn.Sequential(
            nn.Linear(self.d_model, self.d_model//2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model//2, self.d_model)
        )

        # (예: FFT blocks, SCFFTBlock 등)
        self.layer_stack = nn.ModuleList([
            FFTBlock( 
                d_model = self.d_model,
                d_inner = self.d_inner,
                n_head  = self.n_head,
                d_k     = self.d_k,
                d_v     = self.d_v,
                fft_conv1d_kernel_size = self.fft_kernel,
                style_dim = self.style_dim,
                dropout   = self.dropout
            )
            for _ in range(self.n_layers)
        ])

        # 최종 FC: 40채널(고주파)
        self.fc_hi = nn.Linear(self.d_model, 40)


    def forward(self, enc_seq, style_vector, mask=None):
        x = self.prenet(enc_seq)

        slf_attn_mask = None
        if mask is not None:
            B, T, _ = x.shape
            slf_attn_mask = mask.unsqueeze(1).expand(-1, T, -1)

        for layer in self.layer_stack:
            x, _ = layer(x, style_vector, mask=mask, slf_attn_mask=slf_attn_mask)

        mel_high = self.fc_hi(x)
        if mask is not None:
            mel_high = mel_high.masked_fill(mask.unsqueeze(-1), 0.0)

        return mel_high, None


class FFTBlock(nn.Module):
    ''' FFT Block '''
    def __init__(self, d_model,d_inner,
                    n_head, d_k, d_v, fft_conv1d_kernel_size, style_dim, dropout):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.saln_0 = StyleAdaptiveLayerNorm(d_model, style_dim)

        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, fft_conv1d_kernel_size, dropout=dropout)
        self.saln_1 = StyleAdaptiveLayerNorm(d_model, style_dim)

    def forward(self, input, style_vector, mask=None, slf_attn_mask=None):
        # multi-head self attn
        slf_attn_output, slf_attn = self.slf_attn(input, mask=slf_attn_mask)
        slf_attn_output = self.saln_0(slf_attn_output, style_vector)
        if mask is not None:
            slf_attn_output = slf_attn_output.masked_fill(mask.unsqueeze(-1), 0)

        # position wise FF
        output = self.pos_ffn(slf_attn_output)
        output = self.saln_1(output, style_vector)
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)

        return output, slf_attn
    
class SCFFTBlock(nn.Module):
    def __init__(self, d_model, d_hid1, d_hid2, n_head, d_k, d_v, style_dim, dropout, highlow="low"):
        super(SCFFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.ln_0 = nn.LayerNorm(d_model)

        if highlow not in ["low", "high"]:
            raise ValueError(f"Invalid highlow value: {highlow}. Must be 'low' or 'high'.")

        self.highlow = highlow  # high/low 저장

        if highlow == 'low':
            self.pos_ffn = SCPositionwiseFeedForward_low(d_model, d_hid1, d_hid2, dropout=dropout)
        else:
            self.pos_ffn = SCPositionwiseFeedForward_high(d_model, d_hid1, d_hid2, dropout=dropout)

        self.ln_1 = nn.LayerNorm(d_model)

    def forward(self, input, d_w, d_g, d_b, p_w, p_g, p_b, mask=None, slf_attn_mask=None):
        # multi-head self attn
        slf_attn_output, slf_attn = self.slf_attn(input, mask=slf_attn_mask)
        slf_attn_output = self.ln_0(slf_attn_output)
        if mask is not None:
            slf_attn_output = slf_attn_output.masked_fill(mask.unsqueeze(-1), 0)
        
        # ✅ `low`일 때는 추가 인자 없이 호출
        if self.highlow == "low":
            output = self.pos_ffn(slf_attn_output, d_w, d_g, d_b, p_w, p_g, p_b)
        else:
            output = self.pos_ffn(slf_attn_output, d_w, d_g, d_b, p_w, p_g, p_b)  # `high`는 기존 방식 유지

        output = self.ln_1(output)
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)

        return output, slf_attn


    

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, fft_conv1d_kernel_size, dropout=0.1):
        super().__init__()
        self.w_1 = ConvNorm(d_in, d_hid, kernel_size=fft_conv1d_kernel_size[0])
        self.w_2 =  ConvNorm(d_hid, d_in, kernel_size=fft_conv1d_kernel_size[1])

        self.mish = Mish()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        residual = input

        output = input.transpose(1, 2)
        output = self.w_2(self.dropout(self.mish(self.w_1(output))))
        output = output.transpose(1, 2)

        output = self.dropout(output) + residual
        return output
    
class SCPositionwiseFeedForward_high(nn.Module):
    ''' SCCNN and two projection layers '''
    def __init__(self, d_in, d_hid1, d_hid2, dropout=0.1):
        super().__init__()
        self.w_1 = ConvNorm(d_in, d_hid1, kernel_size=1)
        self.w_2 =  ConvNorm(d_hid2, d_in, kernel_size=1)
        self.mish = Mish()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, d_w, d_g, d_b, p_w, p_g, p_b):
        '''
        d_w: (B, in_ch, 1, ker)
        d_g: (B, in_ch)
        d_b: (B, in_ch)
        p_w: (B, out_ch, in_ch, 1)
        p_g: (B, out_ch)
        p_b: (B, out_ch)
        default: in_ch = out_ch = 8, ker = 9
        '''
        residual = input
        log_debug(f"[SCPositionwiseFeedForward] Input shape: {input.shape}")
        
        output = input.transpose(1, 2)
        log_debug(f"[SCPositionwiseFeedForward] Transposed input shape: {output.shape}")
        
        output = self.dropout(self.mish(self.w_1(output)))
        log_debug(f"[SCPositionwiseFeedForward] After w_1 and Mish activation: {output.shape}")
        
        # SC-CNN
        batch = output.size(0)
        kernel_size = d_w.size(-1)
        p = (d_w.size(-1)-1)//2 #padding
        in_ch = d_w.size(1)
        
        # weight normalization
        d_w = nn.functional.normalize(d_w, dim=1)*d_g.unsqueeze(-1).unsqueeze(-1)
        p_w = nn.functional.normalize(p_w, dim=1)*p_g.unsqueeze(-1).unsqueeze(-1)
        log_debug(f"[SCPositionwiseFeedForward] Normalized d_w, p_w shapes: {d_w.shape}, {p_w.shape}")
        
        '''
        # original
        out = []
        for i in range(batch):
          #* Depthwise
          val = nn.functional.conv1d(x[i].unsqueeze(0),
                                          d_w[i],d_b[i],padding=p,groups=in_ch)
          #* Pointwise
          val = nn.functional.conv1d(val,p_w[i],p_b[i])
          out.append(val)
        x = torch.stack(out).squeeze(1)
        '''
        #* Use einsum for acceleration
        x_padded = F.pad(output, (p, p))
        log_debug(f"[SCPositionwiseFeedForward] x_padded shape: {x_padded.shape}")
        
        #* Depthwise
        x = torch.einsum('bctk,bcwk->bct', x_padded.unfold(2, kernel_size, 1), d_w)
        log_debug(f"[SCPositionwiseFeedForward] After depthwise convolution (einsum): {x.shape}")
        
        # Pointwise
        x += d_b.unsqueeze(2)
        x = torch.einsum('bct,bco->bot', x, p_w.squeeze(-1))
        x += p_b.unsqueeze(2)
        log_debug(f"[SCPositionwiseFeedForward] After pointwise convolution (einsum): {x.shape}")
        
         # convolution branch 결과를 활용하여 추가 처리
        x = self.dropout(self.mish(x))
        # x의 shape: [B, out_ch, T] → transpose를 통해 [B, T, out_ch]로 변환
        x = x.transpose(1, 2)
        # w_2 적용 (w_2는 ConvNorm으로, 입력 shape이 [B, d_in, T]를 기대할 수 있음)
        x = x.transpose(1, 2)  # [B, out_ch, T]로 재변환
        x = self.w_2(x)
        x = x.transpose(1, 2)  # 최종 shape: [B, T, d_in]
        
        
        # output = self.dropout(self.mish(output))
        # output = self.w_2(output)
        # log_debug(f"[SCPositionwiseFeedForward] After w_2 projection: {output.shape}")

        # output = output.transpose(1, 2)
        output = self.dropout(x) + residual
        log_debug(f"[SCPositionwiseFeedForward] Final output shape: {output.shape}")
        
        return output
    
class SCPositionwiseFeedForward_low(nn.Module):
    ''' SCCNN and two projection layers (Low-frequency version with Convolution) '''
    def __init__(self, d_in, d_hid1, d_hid2, dropout=0.1):
        super().__init__()
        
        self.w_1 = ConvNorm(d_in, d_hid1, kernel_size=1)  # Linear -> Conv
        self.w_2 = ConvNorm(d_hid2, d_in, kernel_size=1)  # Linear -> Conv
        self.mish = Mish()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, d_w, d_g, d_b, p_w, p_g, p_b):
        '''
        input: [B, T, d_in]
        Apply Depthwise & Pointwise convolution similar to SCPositionwiseFeedForward_high
        '''
        residual = input
        log_debug(f"[SCPositionwiseFeedForward_low] Input shape: {input.shape}")

        # (1) First 1x1 Conv (w_1)
        # input은 [B,T,d_in], → transpose → [B,d_in,T] → conv → [B,d_hid1,T] → transpose back
        output = self.w_1(input.transpose(1, 2)).transpose(1, 2)
        output = self.mish(output)
        output = self.dropout(output)
        log_debug(f"[SCPositionwiseFeedForward_low] After w_1 and Mish activation: {output.shape}")

        # (2) 이제 depthwise를 위해서 [B, C, T] 형태로 다시 만들어줌
        #     (현재 output.shape = [B,T,d_hid1], but d_hid1=8 in your logs)
        #     => transpose(1,2) => [B, d_hid1, T]
        output_c_first = output.transpose(1, 2)  
        #   => shape [B, C, T]  (C=d_hid1=8, T=184 등)

        # (3) padding + unfold(dim=2)
        kernel_size = d_w.size(-1)
        p = (kernel_size - 1) // 2

        # pad last dimension => dim=2
        x_padded = F.pad(output_c_first, (p, p))
        log_debug(f"[SCPositionwiseFeedForward_low] x_padded shape: {x_padded.shape}")

        x_unfolded = x_padded.unfold(2, kernel_size, 1) 
        # => [B, C, (T), K]  (ex: [64, 8, 184, 7])
        log_debug(f"[SCPositionwiseFeedForward_low] x_unfolded shape: {x_unfolded.shape}")

        # (4) Weight normalization
        d_w = nn.functional.normalize(d_w, dim=1) * d_g.unsqueeze(-1).unsqueeze(-1)
        p_w = nn.functional.normalize(p_w, dim=1) * p_g.unsqueeze(-1).unsqueeze(-1)
        log_debug(f"[SCPositionwiseFeedForward_low] Normalized d_w, p_w shapes: {d_w.shape}, {p_w.shape}")

        # expand d_w to match x_unfolded.shape[2]
        d_w = d_w.expand(-1, -1, x_unfolded.shape[2], -1)
        log_debug(f"[SCPositionwiseFeedForward_low] Adjusted d_w shape: {d_w.shape}")

        # (5) Depthwise conv: einsum('bctk, bcwk -> bct')
        x = torch.einsum('bctk, bcwk->bct', x_unfolded, d_w)
        log_debug(f"[SCPositionwiseFeedForward_low] After depthwise convolution (einsum): {x.shape}")

        # Add bias
        x += d_b.unsqueeze(2)

        # (6) Pointwise conv
        #   p_w shape = [B, out_ch, in_ch, 1], out_ch=8, in_ch=8
        #   여기선 einsum('bct,bco->bot')
        x = torch.einsum('bct,bco->bot', x, p_w.squeeze(-1))
        x += p_b.unsqueeze(2)
        log_debug(f"[SCPositionwiseFeedForward_low] After pointwise convolution (einsum): {x.shape}")
        # => shape [B, out_ch, T]

        # (7) 두 번째 1x1 Conv (w_2)
        #   현재 x = [B, out_ch, T],  => transpose back => [B, T, out_ch]
        x_t = x.transpose(1, 2)
        out_2 = self.w_2(x_t.transpose(1, 2)).transpose(1, 2)
        log_debug(f"[SCPositionwiseFeedForward_low] After w_2 projection: {out_2.shape}")

        out_2 = self.dropout(out_2) 
        # residual도 [B,T,d_in], out_2도 [B,T,d_in], shape 동일
        output = out_2 + residual
        log_debug(f"[SCPositionwiseFeedForward_low] Final output shape: {output.shape}")

        return output

class MelStyleEncoder(nn.Module):
    ''' MelStyleEncoder '''
    def __init__(self, config):
        super(MelStyleEncoder, self).__init__()
        self.in_dim = config.n_mel_channels  # e.g. 80
        self.hidden_dim = config.style_hidden
        self.out_dim = config.style_vector_dim
        self.kernel_size = config.style_kernel_size
        self.n_head = config.style_head
        self.dropout = config.dropout

        self.spectral = nn.Sequential(
            LinearNorm(self.in_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout),
            LinearNorm(self.hidden_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout)
        )

        self.temporal = nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )

        self.slf_attn = MultiHeadAttention(self.n_head, self.hidden_dim, 
                                self.hidden_dim//self.n_head, self.hidden_dim//self.n_head, self.dropout) 

        self.fc = LinearNorm(self.hidden_dim, self.out_dim)

    def temporal_avg_pool(self, x, mask=None):
        # x: [B, T, out_dim]
        if mask is None:
            out = torch.mean(x, dim=1)
        else:
            len_ = (~mask).sum(dim=1).unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            x = x.sum(dim=1)
            out = torch.div(x, len_)
        return out

    
    def forward(self, x, mask=None):
        """
        x: [B, T, F], e.g. F=80
        mask: [B, T] (optional)
        """
        log_debug(f"[MelStyleEncoder] Input shape: {x.shape}") #([64, 948, 40]
        
        B, T, F = x.shape
        # 만약 F=40이면, 이미 'half' 크기이므로 더 이상 쪼개지 않는다.
        if F == 40:
            log_debug("[MelStyleEncoder] F=40 => split 생략, 전체 구간을 한 번에 처리합니다.")
            # --------------------------------
            # (A) 기존 '싱글 경로' 처리
            # --------------------------------
            # 1) spectral
            x = self.spectral(x)  # [B, T, hidden_dim]
            log_debug(f"[SinglePath] After spectral: {x.shape}") #[64, 948, 128]

            # 2) temporal conv
            x = x.transpose(1, 2)  # [B, hidden_dim, T]
            x = self.temporal(x)
            x = x.transpose(1, 2)  # [B, T, hidden_dim]
            log_debug(f"[SinglePath] After temporal conv: {x.shape}") #[64, 948, 128

            if mask is not None:
                x = x.masked_fill(mask.unsqueeze(-1), 0)
                log_debug("[SinglePath] Masked fill applied")

            # 3) self-attention
            slf_attn_mask = None
            if mask is not None:
                slf_attn_mask = mask.unsqueeze(1).expand(-1, T, -1)
                log_debug(f"[SinglePath] slf_attn_mask shape: {slf_attn_mask.shape}") #[64, 948, 948]

            x, _ = self.slf_attn(x, mask=slf_attn_mask)
            log_debug(f"[SinglePath] After self-attn: {x.shape}") #[64, 948, 128]

            # 4) fc
            x = self.fc(x)  # [B, T, out_dim]
            log_debug(f"[SinglePath] After fc: {x.shape}") #[64, 948, 128]

            # 5) temporal average pooling => style vector [B, out_dim]
            w = self.temporal_avg_pool(x, mask=mask)
            log_debug(f"[SinglePath] Final style vector shape: {w.shape}") #[64, 128]
            return w

        else:
            # --------------------------------
            # (B) 기존 "채널을 반으로 쪼개는" 로직
            # --------------------------------
            half = F // 2
            x_part1 = x[..., :half]   # [B, T, half]
            x_part2 = x[..., half:]   # [B, T, half]

            log_debug(f"[MelStyleEncoder] Split channels: part1={x_part1.shape}, part2={x_part2.shape}")

            ...
            # 이하 기존의 Part1 / Part2 처리
            # (spectral -> temporal -> attn -> fc -> pooling) 각각 한 뒤
            # w = cat([w1, w2], dim=-1) 등등
            ...
            # 2) Attention에서 쓸 마스크 만들기(시간축)
            slf_attn_mask = None
            if mask is not None:
                slf_attn_mask = mask.unsqueeze(1).expand(-1, T, -1)
                log_debug(f"[MelStyleEncoder] slf_attn_mask shape: {slf_attn_mask.shape}")

                # shape => [B, T, T]
                
            # ------------------
            # Process PART 1
            # ------------------
            log_debug("[MelStyleEncoder] Processing Part1...")
            
            
            # spectral => [B, T, hidden_dim]
            # spectral(x_part1) -> Conv(시간축) -> Self_Attn -> FC -> 평균 풀링
            x1 = self.spectral(x_part1)
            log_debug(f"  [Part1] After spectral: {x1.shape}")
            
            # temporal => [B, hidden_dim, T] -> conv -> [B, hidden_dim, T] -> [B, T, hidden_dim]
            x1 = x1.transpose(1, 2)          # [B, hidden_dim, T]
            x1 = self.temporal(x1)           # ...
            x1 = x1.transpose(1, 2)          # back to [B, T, hidden_dim]
            log_debug(f"  [Part1] After temporal conv: {x1.shape}")
            
            if mask is not None:
                x1 = x1.masked_fill(mask.unsqueeze(-1), 0)
                log_debug("  [Part1] Masked fill applied")

            # self-attention => [B, T, hidden_dim]
            x1, _ = self.slf_attn(x1, mask=slf_attn_mask)
            log_debug(f"  [Part1] After self-attn: {x1.shape}")
            
            # fc => [B, T, out_dim]
            x1 = self.fc(x1)
            log_debug(f"  [Part1] After fc: {x1.shape}")
            
            # pooling => [B, out_dim]
            w1 = self.temporal_avg_pool(x1, mask=mask)
            log_debug(f"  [Part1] style vector: {w1.shape}")
            
            # ------------------
            # Process PART 2 (High freq)
            # ------------------
            log_debug("[MelStyleEncoder] Processing Part2...")
            
            x2 = self.spectral(x_part2)
            log_debug(f"  [Part2] After spectral: {x2.shape}")
            
            x2 = x2.transpose(1, 2)
            x2 = self.temporal(x2)
            x2 = x2.transpose(1, 2)
            log_debug(f"  [Part2] After temporal conv: {x2.shape}")
            
            if mask is not None:
                x2 = x2.masked_fill(mask.unsqueeze(-1), 0)
                log_debug("  [Part2] Masked fill applied")

            x2, _ = self.slf_attn(x2, mask=slf_attn_mask)
            log_debug(f"  [Part2] After self-attn: {x2.shape}")
            
            x2 = self.fc(x2)
            log_debug(f"  [Part2] After fc: {x2.shape}")
            
            w2 = self.temporal_avg_pool(x2, mask=mask)
            log_debug(f"  [Part2] style vector: {w2.shape}")

            # ------------------
            # Combine
            # ------------------
            # 예: concat -> [B, out_dim*2]
            # (3) Combine the two style vectors
            w = torch.cat([w1, w2], dim=-1)
            #w = w[:, :128]
            w = w.view(64, 2, 128)
            w = w.mean(dim=1)
            log_debug(f"[MelStyleEncoder] Final style vector shape: {w.shape}")
            
            return w
 

class Prenet(nn.Module):
    ''' Prenet '''
    def __init__(self, hidden_dim, out_dim, dropout):
        super(Prenet, self).__init__()

        self.convs = nn.Sequential(
            ConvNorm(hidden_dim, hidden_dim, kernel_size=3),
            Mish(),
            nn.Dropout(dropout),
            ConvNorm(hidden_dim, hidden_dim, kernel_size=3),
            Mish(),
            nn.Dropout(dropout),
        )
        self.fc = LinearNorm(hidden_dim, out_dim)

    def forward(self, input, mask=None):
        residual = input
        # convs
        output = input.transpose(1,2)
        output = self.convs(output)
        output = output.transpose(1,2)
        # fc & residual
        output = self.fc(output) + residual

        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)
        return output


class KernelPredictor(nn.Module):
    def __init__(self,
                 style_dim = 128,
                 enc_layer = 4,
                 kernel_size = 9,
                 in_ch = 8,
                 out_ch = 8,
                 ):
        super().__init__()
        
        self.num_layers = enc_layer
        self.ker_size = kernel_size
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        self.d_w_ch = self.num_layers*kernel_size*in_ch
        self.d_g_ch = self.num_layers*in_ch
        self.d_b_ch = self.num_layers*in_ch
        self.p_w_ch = self.num_layers*in_ch*out_ch
        self.p_g_ch = self.num_layers*out_ch
        self.p_b_ch = self.num_layers*out_ch
    
        self.proj = nn.Linear(style_dim, self.d_w_ch + self.d_g_ch + self.d_b_ch + self.p_w_ch + self.p_g_ch + self.p_b_ch)
    
    def forward(self, x):
        '''
        Extract (direction, gain, bias) for two different types of convolutions.
        '''
        batch = x.size(0)
        x = self.proj(x)
        d_w, d_g, d_b, p_w, p_g, p_b = torch.split(x,[self.d_w_ch, self.d_g_ch, self.d_b_ch, self.p_w_ch, self.p_g_ch, self.p_b_ch], dim=1)
        
        d_w = d_w.contiguous().view(batch, self.num_layers, self.in_ch, 1, self.ker_size)
        d_g = d_g.contiguous().view(batch, self.num_layers, self.in_ch)
        d_b = d_b.contiguous().view(batch, self.num_layers, self.in_ch)
        p_w = p_w.contiguous().view(batch, self.num_layers, self.out_ch, self.in_ch, 1)
        p_g = p_g.contiguous().view(batch, self.num_layers, self.out_ch)
        p_b = p_b.contiguous().view(batch, self.num_layers, self.out_ch)

        return (d_w, d_g, d_b, p_w, p_g, p_b)