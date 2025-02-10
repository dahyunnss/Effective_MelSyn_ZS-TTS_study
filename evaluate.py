import torch
from dataloader import prepare_dataloader
import torch.nn.functional as F


def evaluate(args, model, step):    
    # Get dataset
    data_loader = prepare_dataloader(args.val_data_path,"val.txt" ,batch_size=64, shuffle=True)
    
 
    # Get loss function
    Loss = model.get_criterion()
    #Loss = model.module.get_criterion()

    # Evaluation
    mel_l_list = []
    d_l_list = []
    f_l_list = []
    e_l_list = []
    current_step = 0
    for i, batch in enumerate(data_loader):
        # Get Data
        id_ = batch["id"]
        # sid, text, mel_target, D, log_D, f0, energy, \
        #       src_len, mel_len, max_src_len, max_mel_len = model.module.parse_batch(batch)
        sid, text, mel_target, D, log_D, f0, energy, \
            src_len, mel_len, max_src_len, max_mel_len = model.parse_batch(batch)
            
        
    
        with torch.no_grad():
            # Forward
            mel_output, _, _, log_duration_output, f0_output, energy_output, src_mask, mel_mask, out_mel_len = model(
                            text, src_len, mel_target, mel_len, D, f0, energy, max_src_len, max_mel_len)
            
            
            # Cal Loss
            mel_loss, d_loss, f_loss, e_loss = Loss(mel_output,  mel_target, 
                    log_duration_output, log_D, f0_output, f0, energy_output, energy, src_len, mel_len)

            # Logger
            m_l = mel_loss.item()
            d_l = d_loss.item()
            f_l = f_loss.item()
            e_l = e_loss.item()

            mel_l_list.append(m_l)
            d_l_list.append(d_l)
            f_l_list.append(f_l)
            e_l_list.append(e_l)

        current_step += 1            
    
    mel_l = sum(mel_l_list) / len(mel_l_list)
    d_l = sum(d_l_list) / len(d_l_list)
    f_l = sum(f_l_list) / len(f_l_list)
    e_l = sum(e_l_list) / len(e_l_list)

    return mel_l, d_l, f_l, e_l


def evaluate_lo(args, model_lo, step):    
    """
    저주파 전용 모델(SCCNN_Low)에 대해 평가:
      - val_lo.txt 등의 리스트를 이용해 batch를 만든다고 가정
      - batch["mel_target"] shape => [B, T, 40]
    """
    # 1) 데이터로더 준비 (val_lo.txt 등)
    data_loader = prepare_dataloader(args.val_data_path, "val.txt", 
                                     batch_size=64, shuffle=False)
    
    # 2) Loss 함수
    Loss = model_lo.get_criterion()

    # 3) 통계 리스트
    mel_l_list = []
    d_l_list = []
    f_l_list = []
    e_l_list = []

    # 4) 평가 루프
    for i, batch in enumerate(data_loader):
        # parse_batch()를 통해 필요한 텐서들 추출
        sid, text, mel_target_lo, D, log_D, f0, energy, \
            src_len, mel_len, max_src_len, max_mel_len = model_lo.parse_batch(batch)

        with torch.no_grad():
            outputs = model_lo(text, src_len, mel_target_lo, mel_len, D, f0, energy, max_src_len, max_mel_len)
            mel_output_lo, _, _, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _ = outputs


            # 🔥 🔹 `mel_output_lo`의 길이를 `mel_target_lo`에 맞추기
            if mel_output_lo.shape[1] != mel_target_lo.shape[1]:
                mel_output_lo = F.interpolate(
                    mel_output_lo.permute(0, 2, 1),  # [B, 40, T]
                    size=mel_target_lo.shape[1],    # 목표 길이
                    mode="linear",
                    align_corners=False
                ).permute(0, 2, 1)  # 다시 [B, T, 40]으로 변환
            
                
             # ✅ `log_duration_output`이 None이면 더미 값으로 대체
            if log_duration_output is None:
                log_duration_output = torch.zeros_like(mel_target_lo[:, :, 0])  # [B, T]
            if f0_output is None:
                f0_output = torch.zeros_like(mel_target_lo[:, :, 0])  # [B, T]
            if energy_output is None:
                energy_output = torch.zeros_like(mel_target_lo[:, :, 0])  # [B, T]

            # loss 계산
            mel_loss, d_loss, f_loss, e_loss = Loss(
                mel_output_lo, mel_target_lo, 
                log_duration_output[:, :mel_target_lo.shape[1]], log_D[:, :mel_target_lo.shape[1]], 
                f0_output[:, :mel_target_lo.shape[1]], f0[:, :mel_target_lo.shape[1]], 
                energy_output[:, :mel_target_lo.shape[1]], energy[:, :mel_target_lo.shape[1]], 
                src_len, mel_len
            )

            mel_l_list.append(mel_loss.item())
            d_l_list.append(d_loss.item())
            f_l_list.append(f_loss.item())
            e_l_list.append(e_loss.item())


    # 평균 loss
    # mel_l = sum(mel_l_list) / len(mel_l_list) if mel_l_list else 0.0
    # d_l   = sum(d_l_list)   / len(d_l_list)   if d_l_list   else 0.0
    # f_l   = sum(f_l_list)   / len(f_l_list)   if f_l_list   else 0.0
    # e_l   = sum(e_l_list)   / len(e_l_list)   if e_l_list   else 0.0
    
    mel_l = sum(mel_l_list) / len(mel_l_list) if len(mel_l_list) > 0 else 0.0
    d_l = sum(d_l_list) / len(d_l_list) if len(d_l_list) > 0 else 0.0
    f_l = sum(f_l_list) / len(f_l_list) if len(f_l_list) > 0 else 0.0
    e_l = sum(e_l_list) / len(e_l_list) if len(e_l_list) > 0 else 0.0

    return mel_l, d_l, f_l, e_l


def evaluate_hi(args, model_hi, step):    
    """
    고주파 전용 모델(SCCNN_High)에 대해 평가:
      - val_hi.txt 등의 리스트를 이용해 batch를 만든다고 가정
      - batch["mel_target"] shape => [B, T, 40] (hi 영역)
    """
    data_loader = prepare_dataloader(args.val_data_path, "val.txt", 
                                     batch_size=64, shuffle=False)
    
    Loss = model_hi.get_criterion()

    mel_l_list = []
    d_l_list = []
    f_l_list = []
    e_l_list = []

    for i, batch in enumerate(data_loader):
        sid, text, mel_target_hi, D, log_D, f0, energy, \
            src_len, mel_len, max_src_len, max_mel_len = model_hi.parse_batch(batch)

        with torch.no_grad():
            # forward => 모델 추론
            mel_output_hi, _, _, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _ = \
                model_hi(text, src_len, mel_target_hi, mel_len, D, f0, energy, max_src_len, max_mel_len)

            if log_duration_output is None:
                log_duration_output = torch.zeros_like(mel_target_hi[:, :, 0])
            if f0_output is None:
                f0_output = torch.zeros_like(mel_target_hi[:, :, 0])
            if energy_output is None:
                energy_output = torch.zeros_like(mel_target_hi[:, :, 0])
                
            # loss 계산 ([B, T, 40])
            mel_loss, d_loss, f_loss, e_loss = Loss(
                mel_output_hi, mel_target_hi, 
                log_duration_output, log_D, 
                f0_output, f0, 
                energy_output, energy, 
                src_len, mel_len
            )

            mel_l_list.append(mel_loss.item())
            d_l_list.append(d_loss.item())
            f_l_list.append(f_loss.item())
            e_l_list.append(e_loss.item())

    # mel_l = sum(mel_l_list) / len(mel_l_list) if mel_l_list else 0.0
    # d_l   = sum(d_l_list)   / len(d_l_list)   if d_l_list   else 0.0
    # f_l   = sum(f_l_list)   / len(f_l_list)   if f_l_list   else 0.0
    # e_l   = sum(e_l_list)   / len(e_l_list)   if e_l_list   else 0.0
    
    mel_l = sum(mel_l_list) / len(mel_l_list) if len(mel_l_list) > 0 else 0.0
    d_l = sum(d_l_list) / len(d_l_list) if len(d_l_list) > 0 else 0.0
    f_l = sum(f_l_list) / len(f_l_list) if len(f_l_list) > 0 else 0.0
    e_l = sum(e_l_list) / len(e_l_list) if len(e_l_list) > 0 else 0.0

    return mel_l, d_l, f_l, e_l
