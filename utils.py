import os
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt

matplotlib.use("Agg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device(data, device):
    if len(data) == 17:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
            raw_quary_texts,
            quary_texts,
            quary_src_lens,
            max_quary_src_len,
            quary_durations,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        quary_texts = torch.from_numpy(quary_texts).long().to(device)
        quary_src_lens = torch.from_numpy(quary_src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        pitches = torch.from_numpy(pitches).float().to(device)
        energies = torch.from_numpy(energies).to(device)
        durations = torch.from_numpy(durations).long().to(device)
        quary_durations = torch.from_numpy(quary_durations).long().to(device)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
            raw_quary_texts,
            quary_texts,
            quary_src_lens,
            max_quary_src_len,
            quary_durations,
        )

    if len(data) == 10:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            ref_infos,
        ) = data
        
        texts = torch.from_numpy(np.array(texts)).long().to(device)
        src_lens = torch.from_numpy(np.array(src_lens)).to(device)
        mels = torch.from_numpy(np.array(mels)).float().to(device)
        mel_lens = torch.from_numpy(np.array(mel_lens)).to(device)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            ref_infos,
        )
        
        
def process_meta(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        text = []
        name = []
        sid = []
        for line in f.readlines():
            n, t, s = line.strip('\n').split('|')
            name.append(n)
            text.append(t)
            sid.append(s)
        return name, text, sid
    

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def plot_data(data, titles=None, filename=None):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    fig.tight_layout()
    if titles is None:
        titles = [None for i in range(len(data))]
    for i in range(len(data)):
        spectrogram = data[i]
        axes[i][0].imshow(spectrogram, origin='lower')
        axes[i][0].set_aspect(2.5, adjustable='box')
        axes[i][0].set_ylim(0, 80)
        axes[i][0].set_title(titles[i], fontsize='medium')
        axes[i][0].tick_params(labelsize='x-small', left=False, labelleft=False) 
        axes[i][0].set_anchor('W')
    
    plt.savefig(filename, dpi=200)
    plt.close()


# def get_mask_from_lengths(lengths, max_len=None): #gpu
#     batch_size = lengths.shape[0]
#     if max_len is None:
#         max_len = torch.max(lengths).item()

#     ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1) #gpu
#     mask = (ids >= lengths.unsqueeze(1).expand(-1, max_len)) #gpu
 
#     #print("mask",mask.shape)
#     return mask

def get_mask_from_lengths(lengths, max_len=None):
    # 🚨 lengths가 음수 값이 있으면 0으로 변환
    lengths = torch.clamp(lengths, min=0)  # ✅ 잘못된 인자 수정
    lengths = lengths.to(torch.int64)

    # 🚀 lengths가 2D 이상이면 1D로 변환
    if len(lengths.shape) > 1:
        lengths = lengths.view(lengths.shape[0], -1).max(dim=1)[0]  # ✅ 배치에서 최댓값 사용

    #print(f"Filtered lengths (after clamp & squeeze): {lengths.shape}, values: {lengths}")

    # 최대 길이 설정
    if max_len is None:
        max_len = int(torch.max(lengths).item())

    # 🚨 max_len이 0 이하라면 최소 1로 설정
    if max_len <= 0:
        max_len = 1

    #print(f"Final max_len: {max_len}")

    # ✅ ids shape 수정
    ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(lengths.shape[0], -1)

    # ✅ lengths를 1D로 맞춰야 함
    mask = (ids >= lengths.unsqueeze(1))

    return mask

def pad_1D(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])
    return padded


def pad_2D(inputs, maxlen=None):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])
    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0)for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len-batch.size(0)), "constant", 0.0)
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded
    

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, t_path)
