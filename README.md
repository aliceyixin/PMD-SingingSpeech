# PMD-SingingSpeech
This is the author's official PyTorch implementation for ["Disentangled Adversarial Domain Adaptation for Phonation Mode Detection in Singing and Speech"](https://ieeexplore.ieee.org/document/10262362) published in IEEE/ACM Transactions on Audio, Speech, and Language Processing.


## Abstract
Phonation mode detection predicts phonation modes and their temporal boundaries in singing and speech, holding promise for characterizing voice quality and vocal health. However, it is very challenging due to the domain disparities between training data and unannotated real-world recordings. To tackle this problem, we develop a disentangled adversarial domain adaptation network, which adapts the phonation mode detection model with the structure of the convolutional recurrent neural network pre-trained on the source domain to the target domain without phonation mode labels. Based on our curated sung and spoken dataset for phonation mode detection, we demonstrate that the subject and the singing-speech mismatches cause performance decline. By disentangling domain-invariant phonation mode and domain-specific embeddings, our method greatly enhances the effectiveness and explainability of unsupervised adversarial domain adaptation. Experiments show that the performance drop caused by the subject mismatch is alleviated via adaptation, resulting in 44.7% and 6.8% improvement of the F-score for singing and speech, respectively. The singing and speech domain adaptation experiment indicates that a model trained on singing data can be adapted to speech, yielding an F-score of 0.56, commensurate with the F-score of 0.59 achieved using a model exclusively trained on speech data. By further investigating the disentangled embeddings, we find that the phonation mode feature shared by singing and speech is invariant to pitch. These results inspire reliable and versatile applications in voice quality evaluation and paralinguistic information retrieval.

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
@ARTICLE{10262362,
  author={Wang, Yixin and Wei, Wei and Gu, Xiangming and Guan, Xiaohong and Wang, Ye},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={Disentangled Adversarial Domain Adaptation for Phonation Mode Detection in Singing and Speech}, 
  year={2023},
  volume={31},
  number={},
  pages={3746-3759},
  doi={10.1109/TASLP.2023.3317568}}

```
