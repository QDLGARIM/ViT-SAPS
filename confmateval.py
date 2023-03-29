# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 10:11:05 2022

@author: eyxysdht
"""

import numpy as np


def single2confmat(n_cls, prediction, label):
    """Transfer a single prediction and corresponding label to a confuse matrix
    n_cls: Number of classes
    prediction: 2-D array, H*W
    label: 2-D array, H*W"""
    k = (prediction >= 0) & (prediction < n_cls)
    return np.bincount(n_cls * label[k] + prediction[k], minlength=n_cls ** 2).reshape(n_cls, n_cls)

def mbatch2confmats(n_cls, prediction, label):
    """Transfer a minibatch of predictions and corresponding labels to confuse matrices
    n_cls: Number of classes
    prediction: 3-D array, N*H*W
    label: 3-D array, N*H*W"""
    bs = prediction.shape[0]
    confmats = np.zeros([bs, n_cls, n_cls], dtype='int')
    for b in range(bs):
        k = (prediction[b] >= 0) & (prediction[b] < n_cls)
        confmats[b] = np.bincount(n_cls * label[b][k] + prediction[b][k], minlength=n_cls ** 2).reshape(n_cls, n_cls)
    return confmats

def iou_via_confmat(confmat):
    np.seterr(divide="ignore", invalid="ignore")
    iou = np.diag(confmat) / (np.sum(confmat, axis=1) + np.sum(confmat, axis=0) - np.diag(confmat))
    np.seterr(divide="warn", invalid="warn")
    return iou

def mbatchiou_via_confmats(confmats):
    np.seterr(divide="ignore", invalid="ignore")
    [bs, n_cls, _] = confmats.shape
    iou = np.zeros([bs, n_cls])
    for b in range(bs):
        iou[b] = np.diag(confmats[b]) / (np.sum(confmats[b], axis=1) + np.sum(confmats[b], axis=0) - np.diag(confmats[b]))
    np.seterr(divide="warn", invalid="warn")
    return iou
