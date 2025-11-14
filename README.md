# Self‑Attentive Sequential Recommendation (Enriched Features) – PyTorch

# 

# Author: Trong Ho – hdhtrong@gmail.com

# 

# This repository implements self‑attentive sequential recommendation models enriched with user ratings and item categories for improved recommendation accuracy, based on:

# 

# Self‑Attentive Sequential Recommendation Models Enriched with More Features

# ACM DL

# 

# It extends SASRec/TiSASRec by incorporating auxiliary information into input representations, improving sequential recommendation performance on public datasets (e.g., ml‑1m).

# 

# Features

# 

# Self‑attention based sequential recommendation

# 

# Enriched with user ratings \& item categories

# 

# PyTorch v1.6 implementation

# 

# Pretrained models included for inference

# 

# Code structure partially based on/modified from pmixer/TiSASRec.pytorch

# 

# Installation

# git clone <this‑repo‑url>  

# cd <this‑repo>  

# pip install ‑r requirements.txt  

# 

# Quick Start

# Training

# python main.py --dataset=ml‑1m --train\_dir=default --device=cuda  

# 

# Inference with Pretrained Model

# python main.py \\

# &nbsp; --dataset=ml‑1m \\

# &nbsp; --train\_dir=default \\

# &nbsp; --dropout\_rate=0.2 \\

# &nbsp; --device=cuda \\

# &nbsp; --state\_dict\_path='ml‑1m\_default/TiSASRec.epoch=601.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth' \\

# &nbsp; --inference\_only=true \\

# &nbsp; --maxlen=200  

# 

# Notes

# 

# Code based in part on the PyTorch port of TiSASRec: pmixer/TiSASRec.pytorch

# 

# Pretrained models are provided for quick testing

# 

# For detailed explanation of the model architecture and sequential recommendation background, refer to the original paper and referenced repos

# 

# References

# 

# Original TensorFlow repo: JiachengLi1995/TiSASRec

# 

# PyTorch implementation referenced: pmixer/TiSASRec.pytorch

# 

# Paper BibTeX:

# 

# @inproceedings{li2020time,

# &nbsp; title={Time Interval Aware Self‑Attention for Sequential Recommendation},

# &nbsp; author={Li, Jiacheng and Wang, Yujie and McAuley, Julian},

# &nbsp; booktitle={Proceedings of the 13th International Conference on Web Search and Data Mining},

# &nbsp; pages={322‑330},

# &nbsp; year={2020}

# }

