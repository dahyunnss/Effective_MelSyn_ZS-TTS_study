import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]= "0"
import argparse
import random
from preprocessors.libritts import Preprocessor
#from preprocessors.vctk import Preprocessor
import json
from env import AttrDict
import preprocessors.libritts as libritts



def make_train_files(out_dir, datas):    
    random.shuffle(datas)
    
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in datas:
            f.write(m + '\n')


def make_folders(out_dir):
    mel_out_dir = os.path.join(out_dir, "mel")
    if not os.path.exists(mel_out_dir):
        os.makedirs(mel_out_dir, exist_ok=True)
    ali_out_dir = os.path.join(out_dir, "alignment")
    if not os.path.exists(ali_out_dir):
        os.makedirs(ali_out_dir, exist_ok=True)
    f0_out_dir = os.path.join(out_dir, "f0")
    if not os.path.exists(f0_out_dir):
        os.makedirs(f0_out_dir, exist_ok=True)
    energy_out_dir = os.path.join(out_dir, "energy")
    if not os.path.exists(energy_out_dir):
        os.makedirs(energy_out_dir, exist_ok=True)


def main(data_dir, out_dir, libritts):
    libritts.write_metadata(data_dir, out_dir)
    print("data_dir ===",data_dir)
    print("out_dir ===",out_dir)
    make_folders(out_dir)
    datas = libritts.build_from_path(data_dir, out_dir) 
    make_train_files(out_dir, datas)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #LibriTTS
    # parser.add_argument('--data_path', type=str, default='./preprocessed_data/LibriTTS/train-clean-360')
    # parser.add_argument('--output_path', type=str, default='./preprocessed_data/LibriTTS/train-clean-360')
    # parser.add_argument("--config", type=str, default='./configs/config.json', help="path to preprocess.yaml")
    
    parser.add_argument('--data_path', type=str, default='./preprocessed_data/LibriTTS/test-clean')
    parser.add_argument('--output_path', type=str, default='./preprocessed_data/LibriTTS/test-clean')
    parser.add_argument("--config", type=str, default='./configs/config.json', help="path to preprocess.yaml")
    
    #VCTK
    # parser.add_argument('--data_path', type=str, default='./preprocessed_data/VCTK')
    # parser.add_argument('--output_path', type=str, default='./preprocessed_data/VCTK')
    # parser.add_argument("--config", type=str, default='./configs/config_vctk.json', help="path to preprocess.yaml")
    args = parser.parse_args()

    with open(args.config) as f: 
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    preprocessor = Preprocessor(h)
    main(args.data_path, args.output_path, preprocessor)
    
