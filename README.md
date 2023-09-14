# PMD-SingingSpeech
This is the author's official PyTorch implementation for ["Disentangled Adversarial Domain Adaptation for Phonation Mode Detection in Singing and Speech"]().

This paper has been ACCEPTED for publication in the IEEE Transactions on Audio, Speech and Language Processing.


## Abstract

(To be added)

## Prerequisites
Install Anaconda and create the environment with python 3.8.13, pytorch 1.11.0 and cuda 11.3:
```
conda create -n pmd python=3.8.13
conda activate pmd
pip install torch==1.11.0+cu113 torchaudio==0.11.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Our expeiments is built on [SpeechBrain toolkit](https://github.com/speechbrain/speechbrain). To install it, please run the following commands:
```
pip install speechbrain
```

## Usage
We follow the training and evaluation logic of SpeechBrain. Please refer to [SpeechBrain tutorials](https://speechbrain.github.io/).

To run subject domain adaptation experiment:
```
python train_da.py models/DADAN/run_model.yaml
```

To run content domain adaptation experiment:
```
python train_da_ss.py models/DADAN_SS/run_model.yaml
```

## Citation
If you use this paper or codebase in your own work, please cite our paper:

```BibTex
(To be added)
```
