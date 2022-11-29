# Automatic Speech Recognition (ASR)

This repository contains code for training of DeepSpeech2 model, which have been described in the article https://arxiv.org/abs/1512.02595

## Reproduce results

### Install dependencies
```shell
pip install -r ./requirements.txt
```

### Train model 

```shell
python train.py -c hw_asr/configs/config.json
```
