import torch
import numpy as np
import os
import argparse
import librosa
import librosa.display
import re
import json
from string import punctuation
from g2p_en import G2p
import datetime
import matplotlib.pyplot as plt

from models_new_2conv.SCCNN_new_0206_cpu import SCCNN_Low, SCCNN_High
from hifigan.models import Generator as HiFiGAN
import hifigan
from text import text_to_sequence
import audio as Audio
import utils
import time
from tqdm import tqdm
import traceback

import soundfile as sf
from pymcd.mcd import Calculate_MCD
from pesq import pesq


mcd_toolbox = Calculate_MCD(MCD_mode="plain")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_filename(ref_audio):
    return os.path.splitext(os.path.basename(ref_audio))[0]

def wav_to_mel_spectrogram(wav_path, sr=16000, n_mels=80, n_fft=1024, hop_length=256):
    """
    WAV 파일을 멜 스펙트로그램으로 변환합니다.

    Parameters:
        wav_path (str): WAV 파일 경로
        sr (int): 샘플링 레이트
        n_mels (int): 멜 밴드 개수
        n_fft (int): FFT 크기
        hop_length (int): 홉 크기

    Returns:
        np.ndarray: 변환된 멜 스펙트로그램
    """
    y, sr = librosa.load(wav_path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)  # 데시벨 스케일로 변환
    return mel_db



def save_npy_with_original_name(wav_path, output_dir, data):
    """
    원본 WAV 파일과 동일한 이름으로 NumPy 배열을 저장하는 함수.
    
    Parameters:
        wav_path (str): 원본 WAV 파일 경로
        output_dir (str): 저장할 디렉토리
        data (np.ndarray): 저장할 NumPy 데이터
    """
    base_name = extract_filename(wav_path)  # 'audio01'
    npy_path = os.path.join(output_dir, base_name)  # 확장자 없이 저장
    np.save(npy_path, data)  # 'audio01.npy'로 저장됨
    print(f"[INFO] Saved: {npy_path}.npy")


def visualize_mel_spectrograms(base_name, save_path, original_wav_folder):
    """
    생성된 멜-스펙트로그램과 원본 WAV 파일을 비교하여 시각화합니다.
    """
    # 🔹 경로 설정 (base_name 기준)
    generated_npy_path = os.path.join(save_path, f"{base_name}.npy")
    output_image_path = os.path.join(save_path, f"{base_name}_mel_comparison.png")

    # 🔹 1. `.npy` 파일 존재 여부 확인
    if not os.path.exists(generated_npy_path):
        print(f"❌ [ERROR] Generated mel-spectrogram file not found: {generated_npy_path}")
        return

    # 🔹 2. 원본 `WAV` 파일 찾기 (하위 폴더까지 검색)
    original_wav_path = None
    for root, _, files in os.walk(original_wav_folder):
        if f"{base_name}.wav" in files:
            original_wav_path = os.path.join(root, f"{base_name}.wav")
            break

    if original_wav_path is None:
        print(f"❌ [ERROR] Original WAV file not found for {base_name} in {original_wav_folder}")
        return
    
    print(f"✅ [INFO] Matched original_wav_path: {original_wav_path}")

    # 🔹 3. `.npy` 파일 로드
    try:
        generated_mel = np.load(generated_npy_path)
        if generated_mel.ndim == 3:  # 3차원 배열인 경우 squeeze
            generated_mel = np.squeeze(generated_mel)
        print(f"✅ [INFO] Loaded Generated Mel Shape: {generated_mel.shape}")
    except Exception as e:
        print(f"❌ [ERROR] Failed to load .npy file {generated_npy_path}: {e}")
        return

    # 🔹 4. 원본 WAV 파일을 Mel-spectrogram으로 변환
    try:
        original_mel = wav_to_mel_spectrogram(original_wav_path)
        print(f"✅ [INFO] Loaded Original Mel Shape: {original_mel.shape}")
    except Exception as e:
        print(f"❌ [ERROR] Failed to process WAV file {original_wav_path}: {e}")
        return

    # 🔹 5. 길이 맞추기
    min_time_steps = min(generated_mel.shape[1], original_mel.shape[1])
    generated_mel = generated_mel[:, :min_time_steps]
    original_mel = original_mel[:, :min_time_steps]

    # 🔹 6. 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(generated_mel, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title("Generated mel-spectrogram")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Mel Frequency")

    axes[1].imshow(original_mel, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title("Ground-truth mel-spectrogram")
    axes[1].set_xlabel("Time")

    plt.tight_layout()

    # 🔹 7. 이미지 저장 확인 후 저장
    try:
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f"✅ [INFO] Comparison image saved: {output_image_path}")
    except Exception as e:
        print(f"❌ [ERROR] Failed to save image {output_image_path}: {e}")

    plt.show()


def calculate_rtf(audio_length, processing_time):
    return processing_time / audio_length

def calculate_pesq(ref_wav_path, synth_wav_path, sr=16000):
    ref, _ = librosa.load(ref_wav_path, sr=sr)
    synth, _ = librosa.load(synth_wav_path, sr=sr)
    ref = np.asarray(ref * 32768, dtype=np.int16)  # Convert to int16 format
    synth = np.asarray(synth * 32768, dtype=np.int16)  # Convert to int16 format
    return pesq(sr, ref, synth, 'wb')

def create_log_file():
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filename = f"result_Libri100tr_test-clean_cpu_{timestamp}.txt"
    return log_filename


def log_data(log_filename, data):
    with open(log_filename, "a") as file:
        file.write(data + "\n")

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, lexicon_path):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(lexicon_path)

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")
    
    sequence = np.array(text_to_sequence(phones, ['english_cleaners']))
    
    return torch.from_numpy(sequence)


def preprocess_audio(audio_path, sampling_rate, _stft): 
    wav, sample_rate = librosa.load(audio_path, sr=None)
    if sample_rate != sampling_rate: #22050 != 16000
        wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=sampling_rate)
    wav, _ = librosa.effects.trim(wav, top_db=20)
    mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, _stft)
    log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-10, a_max=None))
    return torch.from_numpy(log_mel_spectrogram)


def get_SCCNN_LO_HI(config, checkpoint_path, device):
    """Load both LO and HI SCCNN models from a single checkpoint"""
    # 🔥 체크포인트 파일 확인
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")


    checkpoint = torch.load(checkpoint_path, map_location=device)

    print(f"Checkpoint keys: {checkpoint.keys()}")
    model_lo = SCCNN_Low(config).to(device)
    model_hi = SCCNN_High(config).to(device)
    
    # 🔥 모델 키 존재 여부 확인
    if 'model_lo' not in checkpoint or 'model_hi' not in checkpoint:
        raise KeyError(f"Checkpoint does not contain expected keys. Found keys: {checkpoint.keys()}")

    # 🔥 모델 상태 딕셔너리 키 확인
    print("Model LO keys:", model_lo.state_dict().keys())
    print("Checkpoint LO keys:", checkpoint['model_lo'].keys())

    try:
        model_lo.load_state_dict(checkpoint['model_lo'])
        model_hi.load_state_dict(checkpoint['model_hi'])
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        model_lo.load_state_dict(checkpoint['model_lo'], strict=False)
        model_hi.load_state_dict(checkpoint['model_hi'], strict=False)


    model_lo.eval()
    model_hi.eval()

    return model_lo, model_hi

def get_vocoder(config, device):
    """
    HiFi-GAN Universal 모델(또는 LJSpeech 모델 등)을 불러와서
    PyTorch nn.Module 형태로 반환하는 함수.
    """
    # (1) HiFi-GAN 설정 파일(hifigan/config.json) 불러오기
    config_path = os.path.join("hifigan", "config.json")
    with open(config_path, "r") as f:
        hifigan_config = json.load(f)
        
    hifigan_config = hifigan.AttrDict(hifigan_config)
        
    # (2) HiFi-GAN Generator 초기화
    vocoder = HiFiGAN(hifigan_config).to(device)
    
    # (3) Universal용 가중치 로드 (다른 모델을 쓰려면 경로 바꾸기)
    checkpoint_path = os.path.join("hifigan", "generator_universal.pth.tar")
    checkpoint_dict = torch.load(checkpoint_path, map_location=device)
    vocoder.load_state_dict(checkpoint_dict["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()  # weight_norm 제거 (inference 시 필요)

    return vocoder

def mel2wav(mel, vocoder, device):
    """
    mel: np.array or torch.Tensor
         shape = [T, n_mel_channels] 혹은 [n_mel_channels, T]
    vocoder: HiFi-GAN Generator (get_vocoder에서 불러옴)
    """
    if isinstance(mel, np.ndarray):
        mel = torch.from_numpy(mel)
        
    mel = mel.unsqueeze(0).to(device).float()  # [1, 80, T]
    if mel.dim() == 4:  # ❌ 잘못된 차원: [1, 1, 80, T]
        mel = mel.squeeze(1) 

    with torch.no_grad():
        audio = vocoder(mel)
    audio = audio.squeeze().cpu().numpy()  # -> [audio_length]
    return audio



def synthesize(args, text, model_lo, model_hi, _stft, log_filename,device):  
    print(f"Requested device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")
    
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    ref_audio_files = [os.path.join(root, f) for root, _, files in os.walk(args.ref_audio_dir) for f in files if f.endswith('.wav')]
   
    progress_bar = tqdm(total=len(ref_audio_files), desc="Processing batches", unit="batch", leave=True)

    all_mcd_scores = []
    all_pesq_scores = []
    all_inference_times = []
    all_rtf_scores = []

    for idx, ref_audio in enumerate(ref_audio_files):
        print(f"Processing batch {idx+1}/{len(ref_audio_files)}: {ref_audio}...")
        
        try:
            base_name = os.path.basename(ref_audio)  # 파일명 (확장자 포함) 추출
            base_name = os.path.splitext(base_name)[0]  # 확장자 제거
            
            print(f"🟢 Extracted base_name: {base_name}")  # ✅ 여기서 디버깅

            # 기존: base_name이 text 안에 포함된 문자열을 찾음
            # 변경: 텍스트 파일이 "4992_41806_000032_000001|텍스트" 형식이라면, `split('|')[0]` 사용
            text_entries = [line.split('|')[0] for line in text]  # ✅ base_name만 추출

            matched_text = None
            for txt in text:
                if base_name == txt.split('|')[0]:  # ✅ 정확한 비교
                    matched_text = txt
                    print(f"✅ [INFO] Matched text found: {matched_text}")
                    break  # 첫 번째 매칭된 텍스트 사용

            if not matched_text:
                print("❌ [ERROR] No matching text found! Printing first 5 text entries for debugging:")
                for i, line in enumerate(text[:5]):  # 앞 5개만 출력
                    print(f"   {i+1}. {line}")
                raise ValueError(f"❌ ERROR: No matching text found for {base_name}. Please check the input text file.")

            # # ✅ `text`에서 `ref_audio`의 `base_name`과 일치하는 텍스트 찾기
            # matched_text = None
            # for txt in text:
            #     if base_name in txt:
            #         matched_text = txt
            #         break  # 첫 번째 매칭된 텍스트 사용

            # if not matched_text:
            #     raise ValueError(f"❌ ERROR: No matching text found for {base_name}. Please check the input text file.")

            # print(f"✅ Matched text for {base_name}: {matched_text}")


            ref_mel = preprocess_audio(ref_audio, args.sampling_rate, _stft).transpose(0,1).unsqueeze(0)
            print("ref_mel", ref_mel.shape) #1,80,77
        
            # Extract style vectors
            style_vector_lo = model_lo.get_style_vector(ref_mel).to(device)
            style_vector_hi = model_hi.get_style_vector(ref_mel).to(device)
            print(f"Style Vector Shapes - LO: {style_vector_lo.shape}, HI: {style_vector_hi.shape}")
            
            mcd_scores = []
            pesq_scores = []
            inference_times = []
            rtf_scores = []

            # Forward
            src = preprocess_english(matched_text, args.lexicon_path).unsqueeze(0).to(device)
            src_len = torch.tensor([src.shape[1]], device=device)
            print(f"📝 Text input shape: {src.shape}, src_len: {src_len}")

                # for txt in text:
                #     src = preprocess_english(txt, args.lexicon_path).unsqueeze(0).to(device)
                #     src_len = torch.tensor([src.shape[1]], device=device)
                #     print(f"Text input shape: {src.shape}, src_len: {src_len}")
                    
            cpu_start = time.process_time()
            with torch.no_grad():
                mel_output_lo = model_lo.inference(style_vector_lo, src, src_len)[0]
                mel_output_hi = model_hi.inference(style_vector_hi, src, src_len)[0]
            cpu_end = time.process_time()

            inf_time = cpu_end - cpu_start
            inference_times.append(inf_time)
            
            print(f"mel_output_lo shape: {mel_output_lo.shape}, mel_output_hi shape: {mel_output_hi.shape}")
            
            # ✅ **Zero-padding으로 길이 맞추기**
            # 두 출력의 시간 길이를 맞추기 위해 제로 패딩 적용
            max_T = max(mel_output_lo.shape[1], mel_output_hi.shape[1])
            mel_output_lo = torch.nn.functional.pad(mel_output_lo, (0, 0, 0, max_T - mel_output_lo.shape[1]))
            mel_output_hi = torch.nn.functional.pad(mel_output_hi, (0, 0, 0, max_T - mel_output_hi.shape[1]))
            print(f"Adjusted length with padding: {max_T}")
            
            # ★ 문제 1: HI 분기의 정규화 방식을 변경하여, LO 분기의 평균과 표준편차에 맞추는 모멘트 매칭 수행
            lo_mean = mel_output_lo.mean()
            lo_std = mel_output_lo.std()
            hi_mean = mel_output_hi.mean()
            hi_std = mel_output_hi.std()
            print(f"Before normalization: LO mean = {lo_mean:.4f}, LO std = {lo_std:.4f}; HI mean = {hi_mean:.4f}, HI std = {hi_std:.4f}")
            
            # HI 분기의 값들을 (원래값 - HI 평균)을 (LO 표준편차 / HI 표준편차)로 스케일링한 후, LO 평균을 더한다.
            mel_output_hi = (mel_output_hi - hi_mean) * (lo_std / (hi_std + 1e-5)) + lo_mean
            print(f"After normalization: HI mean = {mel_output_hi.mean():.4f}, HI std = {mel_output_hi.std():.4f}")

            # ★ 문제 2: HI 분기가 주파수 축 방향(마지막 차원)으로 반전되었는지 확인하고, 필요 시 뒤집는다.
            # 첫 번째 주파수 대역의 평균 값과 마지막 대역의 평균 값을 비교한다.
            if mel_output_hi[..., 0].mean() < mel_output_hi[..., -1].mean():
                print("HI spectrum appears reversed. Flipping the HI output along the frequency axis.")
                mel_output_hi = torch.flip(mel_output_hi, dims=[2])
                
            # 두 분기를 채널 차원(마지막 차원)에서 결합하여 80채널 mel 스펙트로그램을 생성한다.
            mel_output = torch.cat([mel_output_lo, mel_output_hi], dim=2)  # shape: [1, T, 80]
            mel_output = mel_output.permute(0, 2, 1)  # 최종 shape: [1, 80, T]
            print(f"Final mel_output shape (for HiFi-GAN): {mel_output.shape}")
            print(f"mel_output before vocoder: min = {mel_output.min()}, max = {mel_output.max()}, mean = {mel_output.mean()}")
                                            
            # 시각화를 위하여 멜 스펙트로그램을 저장 (데시벨 스케일 변환은 plot_spectrogram 내에서 수행됨)
            mel_output_log = mel_output.cpu().numpy()
            mel_output_log = torch.log1p(torch.abs(mel_output))  
            
            # ✅ **최종 파일명 (basename만 사용)**
            wav_save_path = os.path.join(save_path, f"{base_name}.wav")
            npy_save_path = os.path.join(save_path, f"{base_name}.npy")
            np.save(npy_save_path, mel_output.cpu().numpy()) 
            print(f"📝 npy_save_path: {npy_save_path}") 
            
            img_save_path = os.path.join(save_path, f"{base_name}_mel_comparison.png")
            print(f"📝 img_save_path: {img_save_path}")
            
            if os.path.exists(npy_save_path):
                visualize_mel_spectrograms(base_name, save_path, args.ref_audio_dir)
                #base_name, save_path, args.ref_audio_dir
            else:
                print(f"❌ [ERROR] .npy file not found: {npy_save_path}")

                
            # ✅ HiFi-GAN vocoder를 통해 파형(waveform) 복원
            wav_output = mel2wav(mel_output.cpu().numpy(), vocoder, device)

            # ✅ WAV 파일 저장 (base_name 유지)
            sf.write(wav_save_path, wav_output, args.sampling_rate)
            print(f"[INFO] Saved waveform: {wav_save_path}")
            #7176_92135_000040_000001_260_123440.wav

            # Calculate RTF
            audio_length = len(wav_output) / args.sampling_rate
            rtf = calculate_rtf(audio_length, inf_time)
            rtf_scores.append(rtf)
            
            # Calculate MCD and PESQ
            synthesized_wav_path = wav_save_path
            reference_wav_path = ref_audio
            
            mcd_value = mcd_toolbox.calculate_mcd(reference_wav_path, synthesized_wav_path)
            mcd_scores.append(mcd_value)

            pesq_value = calculate_pesq(reference_wav_path, synthesized_wav_path, args.sampling_rate)
            pesq_scores.append(pesq_value)
    
            log_data(log_filename, f"{txt.split('|')[0]}: MCD Score = {mcd_value:}, PESQ Score = {pesq_value:}, RTF = {rtf:}")
            log_data(log_filename, f"Inference Time: {inf_time:} seconds")

             # ✅ 전체 리스트에 추가 (최종 평균 계산을 위해)
            all_mcd_scores.append(mcd_value)
            all_pesq_scores.append(pesq_value)
            all_inference_times.append(inf_time)
            all_rtf_scores.append(calculate_rtf(len(wav_output) / args.sampling_rate, inf_time))

            # tqdm 진행 바 업데이트
            progress_bar.update(1)
            
        except Exception as e:
            print(f"❌ Error processing {ref_audio}: {e}")
            traceback.print_exc()
            break
            
    if all_mcd_scores or all_pesq_scores or all_inference_times or all_rtf_scores:
        average_mcd = sum(all_mcd_scores) / len(all_mcd_scores) if all_mcd_scores else 0
        average_pesq = sum(all_pesq_scores) / len(all_pesq_scores) if all_pesq_scores else 0
        avg_inf_time = sum(all_inference_times) / len(all_inference_times) if all_inference_times else 0
        average_rtf = sum(all_rtf_scores) / len(all_rtf_scores) if all_rtf_scores else 0

        log_data(log_filename, f"Average MCD Score: {average_mcd:.7f}, Average PESQ Score: {average_pesq:.7f}, Average RTF: {average_rtf:.7f}")
        log_data(log_filename, f"Average Inference Time:: {avg_inf_time:.7f} seconds")
 
        print('✅ All processing done!')

    progress_bar.close()

if __name__ == "__main__":
    from audio.stft import TacotronSTFT
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, 
        help="Path to the pretrained model")
    parser.add_argument('--config', default='configs/config2.json')
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--ref_audio_dir", type=str, required=True,
        help="path to the directory containing reference speech audio samples")
    parser.add_argument("--text_file", type=str, required=True,
        help="path to the text file containing text to synthesize")
    parser.add_argument("--lexicon_path", type=str, default='lexicon/librispeech-lexicon.txt')
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run the model on (cpu or cuda)",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    
    with open(args.config) as f:
        data = f.read()
    json_config = json.loads(data)
    config = utils.AttrDict(json_config)

    # Get model
    model_lo, model_hi = get_SCCNN_LO_HI(config, args.checkpoint_path, device)
    print(f"Model lo device: {next(model_lo.parameters()).device}")
    print(f"Model hi device: {next(model_hi.parameters()).device}")
 
    # Load vocoder
    vocoder = get_vocoder(config, device)

    _stft = Audio.stft.TacotronSTFT(
                config.filter_length,
                config.hop_length,
                config.win_length,
                config.n_mel_channels,
                config.sampling_rate,
                config.mel_fmin,
                config.mel_fmax)

    log_filename = create_log_file()
    print(f"Log file created: {log_filename}")
    
    print(f"Reading text from {args.text_file}")
    with open(args.text_file, 'r') as file:
        args.text = file.readlines()
        args.text = [line.strip() for line in args.text if line.strip()]  

    args.sampling_rate = config.sampling_rate
    print("Starting synthesis...")
    synthesize(args, args.text, model_lo, model_hi, _stft, log_filename,device)
