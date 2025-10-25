"""Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com)

-----------------------------

train.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from options import Options
from lib.data import load_data
from lib.timegan import TimeGAN
from lib.timegan import BaseModel
import matplotlib.pyplot as plt
import numpy as np


def train():
    """ Training
    """

    # ARGUMENTS
    opt = Options().parse()

    # LOAD DATA
    ori_data = load_data(opt)

    # LOAD MODEL
    model = TimeGAN(opt, ori_data)

    # TRAIN MODEL
    model.train()
    
    plot_losses(BaseModel.loss_history)
    
def smooth(values, alpha=0.95):
    """ Exponential smoothing to reduce noise """
    smoothed = []
    v = 0
    for x in values:
        v = alpha * v + (1 - alpha) * x
        smoothed.append(v)
    return smoothed


def plot_losses(loss_history, smooth_curves=True):
    plt.figure(figsize=(14, 6))

    for key, values in loss_history.items():
        if len(values) == 0:
            continue  # skip empty series

        if smooth_curves:
            plt.plot(smooth(values), label=f"{key.upper()} (smoothed)", alpha=0.9)
        else:
            plt.plot(values, label=key.upper(), alpha=0.7)

    plt.title("✅ TimeGAN Training Losses", fontsize=16)
    plt.xlabel("Training steps", fontsize=12)
    plt.ylabel("Loss value", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    train()
