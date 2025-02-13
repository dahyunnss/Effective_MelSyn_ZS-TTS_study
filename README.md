# Effective_MelSyn_ZS-TTS_study
This repository is made for the "A Study on Effective Mel-Spectrogram Synthesis using Multiple Speakers Sample for Zero-shot Text-To-Speech Task" based on "SC-CNN: Effective Speaker Conditioning Method for Zero-Shot Multi-Speaker Text-to-Speech Systems" from IEEE Signal Processing Letters 2023. [Official Repository](https://github.com/hcy71o/SC-CNN/tree/SC-StyleSpeech) [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10129023)


## Requirements
You can install the Python dependencies with
```
pip install -r requirements.txt
```

## Dataset
The supported datasets are

- [LibriTTS](https://research.google/tools/datasets/libri-tts/): a multi-speaker English dataset containing 585 hours of speech by 2456 speakers.
- (will be added more)


## Inference
Run synthesize.py for inference process:
```
python3 synthesize_proposed_cpu2.py --checkpoint_path output_proposed_low_high_2conv2/ckpt/checkpoint_180000.pth.tar --config ./configs/config2.json --save_path output_proposed_low_high_2conv2/results/cpu/Libri100tr_test-clean --ref_audio_dir DUMMY2/LibriTTS/test-clean2 --text_file preprocessed_data/LibriTTS/test-clean/test.txt --lexicon_path ./lexicon/librispeech-lexicon.txt --device cpu
```
The generated utterances will be put in ``output/result/``. Your synthesized speech will have `ref_audio`'s style.
