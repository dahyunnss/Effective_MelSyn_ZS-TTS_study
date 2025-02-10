import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import json
import time
from models_new_2conv.SCCNN_new_0206 import SCCNN_Low, SCCNN_High
from dataloader import prepare_dataloader
from optimizer import ScheduledOptim
from evaluate2 import evaluate_lo, evaluate_hi
import utils
import torch.nn.functional as F 
        
def log_debug(message, log_path):
    print(message)
    with open(os.path.join(log_path, "debug_log.txt"), "a") as f:
        f.write(message + "\n")
        
def load_dual_checkpoint(checkpoint_path, model_lo, model_hi, optimizer):
    """두 모델(LO/HI)을 모두 로드하기 위한 유틸함수"""
    assert os.path.isfile(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    print(f"Starting from checkpoint '{checkpoint_path}'")
    ckpt_dict = torch.load(checkpoint_path)

    # model_lo
    if 'model_lo' in ckpt_dict:
        model_lo.load_state_dict(ckpt_dict['model_lo'])
        print("Lo-model state loaded!")
    # model_hi
    if 'model_hi' in ckpt_dict:
        model_hi.load_state_dict(ckpt_dict['model_hi'])
        print("Hi-model state loaded!")
    # optimizer
    if 'optimizer' in ckpt_dict:
        optimizer.load_state_dict(ckpt_dict['optimizer'])
        print("Optimizer loaded!")
    # step
    current_step = ckpt_dict.get('step', 0) + 1
    return model_lo, model_hi, optimizer, current_step


def main(args, c):
    """Dual Training: Low/High 모델 동시에 로스 계산 -> 합산 -> 한 번에 backward"""

    log_path = os.path.join(args.save_path, 'log', "Libri100tr_dev-clean")
    os.makedirs(log_path, exist_ok=True)
    
    # 1) Define model (both low & high)
    model_lo = SCCNN_Low(c).cuda()
    model_hi = SCCNN_High(c).cuda()
    #model = nn.DataParallel(model) #DataParallel
    print("Model LO + Model HI instantiated.")
    
    param_lo = utils.get_param_num(model_lo)
    param_hi = utils.get_param_num(model_hi)
    print(f"[LO] # params: {param_lo}, [HI] # params: {param_hi}")
    log_debug(f"LO params={param_lo}, HI params={param_hi}", log_path)

    
    # 2) Optimizer & Loss
    #   - 두 모델 파라미터를 모두 합쳐서 하나의 optimizer로 관리
    params_all = list(model_lo.parameters()) + list(model_hi.parameters())
    optimizer = torch.optim.Adam(params_all, betas=c.betas, eps=c.eps)

    # Loss는 StyleSpeechLoss (혹은 원하는 Loss) 각각
    Loss_lo = model_lo.get_criterion()
    Loss_hi = model_hi.get_criterion()

    print("Optimizer + 2 Loss (lo, hi) ready.")

    # 3) DataLoader
    try:
        log_debug("Preparing dual-batch data loader...", log_path)
        # (예) train_dual.txt 는 mel_target_lo, mel_target_hi 등을 포함
        data_loader = prepare_dataloader(args.data_path, "train.txt",
                                         shuffle=True, batch_size=c.batch_size)
    except Exception as e:
        log_debug(f"Error loading data loader: {str(e)}", log_path)
        raise
    
   
    # 4) Load checkpoint if needed
    current_step = 0
    if args.checkpoint_path is not None:
        model_lo, model_hi, optimizer, current_step = load_dual_checkpoint(
            args.checkpoint_path, model_lo, model_hi, optimizer
        )
        print(f"Dual-model checkpoint loaded at step={current_step}")
    else:
        print("\n*** Start new dual-model training ***\n")
        current_step = 0
    
    # 5) Prepare path
    ckpt_path = os.path.join(args.save_path, "ckpt")
    os.makedirs(ckpt_path, exist_ok=True)

    scheduled_optim = ScheduledOptim(optimizer, c.decoder_hidden, c.n_warm_up_step, current_step)
    logger = SummaryWriter(os.path.join(log_path, "board"))

    # 8) Synthesis directory
    synth_path = os.path.join(args.save_path,'synth',"Libri100tr_dev-clean") #Libri360tr_dev-clean
    os.makedirs(synth_path, exist_ok=True)
    
    # 9) Start timer
    start_time = time.time()

    # 10) Training
    model_lo.train()
    model_hi.train()
    
    while current_step < args.max_iter:        
        # Get Training Loader
        for idx, batch in enumerate(data_loader): 
            if current_step == args.max_iter:
                break
            
            # KeyError 방지 및 디버깅
            if "mel_target_lo" not in batch:
                raise KeyError(f"'mel_target_lo' not found in batch. Available keys: {batch.keys()}")

            if "mel_target_hi" not in batch:
                raise KeyError(f"'mel_target_hi' not found in batch. Available keys: {batch.keys()}")

            
            # (A) Batch parsing
            text    = torch.from_numpy(batch["text"]).long().cuda()
            mel_lo  = torch.from_numpy(batch["mel_target_lo"]).float().cuda()  # [B,T,40]
            mel_hi  = torch.from_numpy(batch["mel_target_hi"]).float().cuda()  # [B,T,40]
            D       = torch.from_numpy(batch["D"]).long().cuda()
            log_D   = torch.from_numpy(batch["log_D"]).float().cuda()
            f0      = torch.from_numpy(batch["f0"]).float().cuda()
            energy  = torch.from_numpy(batch["energy"]).float().cuda()
            src_len = torch.from_numpy(batch["src_len"]).long().cuda()
            mel_len = torch.from_numpy(batch["mel_len"]).long().cuda()
            
            # 🔥 채널 크기 맞추기 (필수 수정)
            mel_lo = mel_lo[:, :, :40]  # 저주파 영역만 추출
            mel_hi = mel_hi[:, :, 40:]  # 고주파 영역만 추출
            
            # Forward
            scheduled_optim.zero_grad()
            
            # Move all tensors to the same device
            device = next(model_lo.parameters()).device
            text, mel_lo, D, log_D, f0, energy, src_len, mel_len = \
                [x.to(device) if isinstance(x, torch.Tensor) else x for x in 
                [text, mel_lo, D, log_D, f0, energy, src_len, mel_len]]
                
            
            # (B) Forward LO model
            mel_pred_lo, src_out_lo, style_lo, d_pred_lo, p_pred_lo, e_pred_lo, src_mask_lo, mel_mask_lo, _ = \
                model_lo(text, src_len, mel_hi, mel_len, D, f0, energy)
            # lo_out = model_lo(
            #     text, src_len, mel_lo, mel_len, D, f0, energy
            # )
            
            # Loss LO
            mel_loss_lo, d_loss_lo, f_loss_lo, e_loss_lo = Loss_lo(
                mel_pred_lo, mel_lo,
                d_pred_lo, log_D,
                p_pred_lo, f0,
                e_pred_lo, energy,
                src_len, mel_len
            )

            #mel_pred_lo, _, _, d_pred_lo, p_pred_lo, e_pred_lo, _, _, _ = lo_out

            # 수정된 Low 모델의 forward 함수는 아래와 같이 반환:
            # (mel_prediction, src_embedded, style_vector, d_prediction, p_prediction, e_prediction, src_mask, mel_mask, mel_len)
            #mel_pred_lo, _, _, d_pred_lo, p_pred_lo, e_pred_lo, src_mask_lo, mel_mask_lo, mel_len_out = lo_out
                        
            # 🔥 🔹 `mel_pred_lo`와 `mel_lo`의 길이를 맞추기
            # if mel_pred_lo.shape[1] != mel_lo.shape[1]:
            #     #log_debug(f"[Fix] Adjusting mel_pred_lo shape: {mel_pred_lo.shape} -> {mel_lo.shape[1]}", log_path) 
            #     mel_pred_lo = mel_pred_lo[:, :mel_lo.shape[1], :]
                
            # if mel_pred_lo.shape[1] != mel_lo.shape[1]:
            #     mel_pred_lo = F.interpolate(
            #         mel_pred_lo.permute(0, 2, 1), size=mel_lo.shape[1], mode="linear", align_corners=False
            #     ).permute(0, 2, 1)

            # if d_pred_lo is None:
            #     # Lo model skip dur/f0/energy
            #     #mel_pred_lo = F.interpolate(mel_pred_lo.permute(0, 2, 1), size=mel_lo.shape[1], mode="linear", align_corners=False).permute(0, 2, 1)

            #     mel_loss_lo = F.l1_loss(mel_pred_lo, mel_lo)
            #     d_loss_lo = 0.0
            #     f_loss_lo = 0.0
            #     e_loss_lo = 0.0

            # else:
            #     # normal
            #     mel_loss_lo, d_loss_lo, f_loss_lo, e_loss_lo = Loss_lo(
            #     mel_pred_lo, mel_lo,
            #     d_pred_lo[:, :mel_lo.shape[1]], log_D[:, :mel_lo.shape[1]],
            #     p_pred_lo[:, :mel_lo.shape[1]], f0[:, :mel_lo.shape[1]],
            #     e_pred_lo[:, :mel_lo.shape[1]], energy[:, :mel_lo.shape[1]],
            #     src_len, mel_len
            # )
            total_lo = mel_loss_lo + d_loss_lo + f_loss_lo + e_loss_lo
            

            # (C) Forward HI model
            mel_pred_hi, src_out_hi, style_hi, d_pred_hi, p_pred_hi, e_pred_hi, src_mask_hi, mel_mask_hi, _ = \
                model_hi(text, src_len, mel_hi, mel_len, D, f0, energy)
            # Loss HI
            mel_loss_hi, d_loss_hi, f_loss_hi, e_loss_hi = Loss_hi(
                mel_pred_hi, mel_hi,
                d_pred_hi, log_D,
                p_pred_hi, f0,
                e_pred_hi, energy,
                src_len, mel_len
            )
            total_hi = mel_loss_hi + d_loss_hi + f_loss_hi + e_loss_hi
            
      
            # (D) total_loss
            total_loss = total_lo + total_hi
            # Backward
            total_loss.backward()
            # Clipping gradients to avoid gradient explosion
            nn.utils.clip_grad_norm_(params_all, c.grad_clip_thresh)
            scheduled_optim.step_and_update_lr()

            
            if current_step % args.std_step == 0 and current_step != 0:    
                t_l = total_loss.item()
                
                # lo
                lo_m_val = mel_loss_lo.item()
                lo_d_val = d_loss_lo.item() if isinstance(d_loss_lo, torch.Tensor) else d_loss_lo
                lo_f_val = f_loss_lo.item() if isinstance(f_loss_lo, torch.Tensor) else f_loss_lo
                lo_e_val = e_loss_lo.item() if isinstance(e_loss_lo, torch.Tensor) else e_loss_lo

                
                # hi
                hi_m_val = mel_loss_hi.item()
                hi_d_val = d_loss_hi.item()
                hi_f_val = f_loss_hi.item()
                hi_e_val = e_loss_hi.item()
                
                msg = (
                    f"Step [{current_step}/{args.max_iter}]"
                    f"\n LO => Mel Loss: {lo_m_val:.4f}, Duration Loss:{lo_d_val:.4f}, F0 Loss:{lo_f_val:.4f}, Energy Loss:{lo_e_val:.4f}"
                    f"\n HI => Mel Loss: {hi_m_val:.4f}, Duration Loss:{hi_d_val:.4f}, F0 Loss:{hi_f_val:.4f}, Energy Loss:{hi_e_val:.4f}"
                    f"\n Total Loss: {t_l:.4f}\n Mel Loss: {lo_m_val + hi_m_val:.4f},\n Duration Loss: {lo_d_val + hi_d_val:.4f}, F0 Loss: {lo_f_val + hi_f_val:.4f}, Energy Loss: {lo_e_val + hi_e_val:.4f}"
                )
                    
                print(msg +"\n")  
                with open(os.path.join(log_path, "log.txt"), "a") as f_log:
                    f_log.write(msg + "\n")
                
                #tensorboard
                if current_step % args.log_step == 0:
                    logger.add_scalar('Train/LO_mel_loss', lo_m_val, current_step)
                    logger.add_scalar('Train/LO_dur_loss', lo_d_val, current_step)
                    logger.add_scalar('Train/LO_f0_loss', lo_f_val, current_step)
                    logger.add_scalar('Train/LO_energy_loss', lo_e_val, current_step)

                    logger.add_scalar('Train/HI_mel_loss', hi_m_val, current_step)
                    logger.add_scalar('Train/HI_dur_loss', hi_d_val, current_step)
                    logger.add_scalar('Train/HI_f0_loss', hi_f_val, current_step)
                    logger.add_scalar('Train/HI_energy_loss', hi_e_val, current_step)

                    logger.add_scalar('Train/Total_mel_loss', lo_m_val + hi_m_val, current_step)
                    logger.add_scalar('Train/Total_dur_loss', lo_d_val + hi_d_val, current_step)
                    logger.add_scalar('Train/Total_f0_loss', lo_f_val + hi_f_val, current_step)
                    logger.add_scalar('Train/Total_energy_loss', lo_e_val + hi_e_val, current_step)
            
            # Save Checkpoint
            if current_step % args.save_step == 0 and current_step != 0:
                save_name = os.path.join(ckpt_path, f"checkpoint_{current_step}.pth.tar")
                torch.save({
                    'model_lo': model_lo.state_dict(),
                    'model_hi': model_hi.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': current_step
                }, save_name)
                print("*** Save Checkpoint ***")
                print(f"Save model at step {save_name}...\n")

            # ---------------------------------------------------
            # (G) Synthesis example
            # ---------------------------------------------------
            #train중 출력된 mel-spectrogram 생성
            if current_step % args.synth_step == 0 and current_step != 0:
                # sample idx=0
                b0_len = mel_len[0].item()

                # LO
                gt_lo = mel_lo[0, :b0_len].detach().cpu().transpose(0, 1)     # => [T,40]
                pred_lo = mel_pred_lo[0, :b0_len].detach().cpu().transpose(0, 1)
                utils.plot_data(
                    [pred_lo.numpy(), gt_lo.numpy()],
                    ["Synthesized Spectrogram-LO", "Ground-Truth Spectrogram-LO"],
                    filename=os.path.join(args.save_path, f"synth/step_{current_step}_lo.png")
                )

                # HI
                gt_hi = mel_hi[0, :b0_len].detach().cpu().transpose(0, 1)     # => [T,40]
                pred_hi = mel_pred_hi[0, :b0_len].detach().cpu().transpose(0, 1)
                utils.plot_data(
                    [pred_hi.numpy(), gt_hi.numpy()],
                    ["Synthesized Spectrogram-HI", "Ground-Truth Spectrogram-HI"],
                    filename=os.path.join(args.save_path, f"synth/step_{current_step}_hi.png")
                )

                print(f"** Synth spectrograms at step {current_step}...\n")

            # ---------------------------------------------------
            # (H) Validation
            # ---------------------------------------------------
            
            if current_step % args.eval_step == 0 and current_step != 0:
                model_lo.eval()
                model_hi.eval()
                with torch.no_grad():
                    # Evaluate LO
                    lo_m_val_eval, lo_d_val_eval, lo_f_val_eval, lo_e_val_eval = evaluate_lo(
                        args, model_lo, current_step
                    )
                    # Evaluate HI
                    hi_m_val_eval, hi_d_val_eval, hi_f_val_eval, hi_e_val_eval = evaluate_hi(
                        args, model_hi, current_step
                    )

                    total_m_val_eval = lo_m_val_eval + hi_m_val_eval
                    total_d_val_eval = lo_d_val_eval + hi_d_val_eval
                    total_f_val_eval = lo_f_val_eval + hi_f_val_eval
                    total_e_val_eval = lo_e_val_eval + hi_e_val_eval
                    
                    
                    msg = (
                        "*** Validation Results ***\n"
                        f"Step [{current_step}/{args.max_iter}]"
                        f"\nLO => Mel: {lo_m_val_eval:.4f}, Duration: {lo_d_val_eval:.4f}, "
                        f"F0: {lo_f_val_eval:.4f}, Energy: {lo_e_val_eval:.4f}"
                        f"\nHI => Mel: {hi_m_val_eval:.4f}, Duration: {hi_d_val_eval:.4f}, "
                        f"F0: {hi_f_val_eval:.4f}, Energy: {hi_e_val_eval:.4f}"
                        f"\nTotal => Mel: {total_m_val_eval:.4f}, Duration: {total_d_val_eval:.4f}, "
                        f"F0: {total_f_val_eval:.4f}, Energy: {total_e_val_eval:.4f}"
                    )
                    print(msg +"\n")  
                    with open(os.path.join(log_path, "eval.txt"), "a") as f_log:
                        f_log.write(msg + "\n")
                        
                    logger.add_scalar('Validation/LO_mel_loss', lo_m_val_eval, current_step)
                    logger.add_scalar('Validation/LO_duration_loss', lo_d_val_eval, current_step)
                    logger.add_scalar('Validation/LO_f0_loss', lo_f_val_eval, current_step)
                    logger.add_scalar('Validation/LO_energy_loss', lo_e_val_eval, current_step)

                    logger.add_scalar('Validation/HI_mel_loss', hi_m_val_eval, current_step)
                    logger.add_scalar('Validation/HI_duration_loss', hi_d_val_eval, current_step)
                    logger.add_scalar('Validation/HI_f0_loss', hi_f_val_eval, current_step)
                    logger.add_scalar('Validation/HI_energy_loss', hi_e_val_eval, current_step)

                    logger.add_scalar('Validation/Total_mel_loss', total_m_val_eval, current_step)
                    logger.add_scalar('Validation/Total_duration_loss', total_d_val_eval, current_step)
                    logger.add_scalar('Validation/Total_f0_loss', total_f_val_eval, current_step)
                    logger.add_scalar('Validation/Total_energy_loss', total_e_val_eval, current_step)

                model_lo.train()
                model_hi.train()

            current_step += 1 
            
    # End timer and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    log_debug(f"Training completed in {int(hours):02}:{int(minutes):02}:{int(seconds):02}", log_path)
    log_debug(f"Final step: {current_step}", log_path)
    
    # Final save
    torch.save({
        'model_lo': model_lo.state_dict(),
        'model_hi': model_hi.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': current_step
    }, os.path.join(ckpt_path, f'checkpoint_last_{current_step}.pth.tar'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='preprocessed_data/LibriTTS/train-clean-100')
    #parser.add_argument('--data_path', default='/userHome/userhome2/dahyun/SC-CNN/preprocessed_data/VCTK')
    parser.add_argument('--save_path', default='output_proposed_low_high_2conv2')
    parser.add_argument('--config', default='configs/config2.json')
    #parser.add_argument('--config', default='configs/config_vctk.json')
    parser.add_argument('--val_data_path', default='preprocessed_data/LibriTTS/dev-clean')
    parser.add_argument('--max_iter', default=200000, type=int) #200000 #400000
    parser.add_argument('--save_step', default=10000, type=int) #2000 #10000
    parser.add_argument('--synth_step', default=2000, type=int) #1000 #2000
    parser.add_argument('--eval_step', default=2000, type=int) #1000 #2000
    parser.add_argument('--test_step', default=10000, type=int)#1000 #1000
    parser.add_argument('--log_step', default=100, type=int) #100 #100
    parser.add_argument('--std_step', default=100, type=int) #100 #10
    parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to the pretrained model') 

    args = parser.parse_args()
    torch.backends.cudnn.enabled = True

    with open(args.config) as f:
        data = f.read()
    json_config = json.loads(data)
    config = utils.AttrDict(json_config)
    utils.build_env(args.config, 'config.json', args.save_path)

    main(args, config)