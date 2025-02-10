import argparse
import preprocessors.libritts as libritts
import preprocessors.vctk as vctk

def main(data_path, sr):
    libritts.prepare_align_and_resample(data_path, sr)
    #vctk.prepare_align_and_resample(data_path, sr)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/userHome/userhome2/dahyun/SC-CNN/raw_data/LibriTTS/train-clean-100')
    #parser.add_argument('--data_path', type=str, default='/userHome/userhome2/dahyun/SC-CNN/preprocessed_data/VCTK')
    parser.add_argument('--resample_rate', '-sr', type=int, default=16000)

    args = parser.parse_args()

    main(args.data_path, args.resample_rate)
