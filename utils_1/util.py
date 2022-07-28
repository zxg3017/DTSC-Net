# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import pickle
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import torch
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
import random
import networks

def load_model(path):
    """Loads model and return it without DataParallel table."""
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)

        # size of the top layer
        N = checkpoint['state_dict']['top_layer.bias'].size()

        # build skeleton of the model
        sob = 'sobel.0.weight' in checkpoint['state_dict'].keys()
        model = models.__dict__[checkpoint['arch']](sobel=sob, out=int(N[0]))

        # deal with a dataparallel table
        def rename_key(key):
            if not 'module' in key:
                return key
            return ''.join(key.split('.module'))

        checkpoint['state_dict'] = {rename_key(key): val
                                    for key, val
                                    in checkpoint['state_dict'].items()}

        # load weights
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded")
    else:
        model = None
        print("=> no checkpoint found at '{}'".format(path))
    return model


class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        size_per_pseudolabel = int(self.N / len(self.images_lists)) + 1
        res = np.zeros(size_per_pseudolabel * len(self.images_lists))

        for i in range(len(self.images_lists)):
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res[i * size_per_pseudolabel: (i + 1) * size_per_pseudolabel] = indexes

        np.random.shuffle(res)
        return res[:self.N].astype('int')

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return self.N


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr


class Logger():
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)


def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        posmask = img_gt[b].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
            sdf[boundary==1] = 0
            normalized_sdf[b] = sdf
            # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
            # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf


def augmentation(volume, aug_factor):
    # volume is numpy array of shape (C, D, H, W)
    return volume + aug_factor * np.clip(np.random.randn(*volume.shape) * 0.1, -0.2, 0.2).astype(np.float32)


def mix_match(X, U, eval_net, K, T, alpha, mixup_mode, aug_factor):
    """
    参考论文 MixMatch: A Holistic Approach to Semi-Supervised Learning
    """
    # X is labeled data of size BATCH_SIZE, and U is unlabeled data
    # X is list of tuples (data, label), and U is list of data
    # where data and label are of shape (C, D, H, W), numpy array. C of data is 1 and C of label is 2 (one hot)
    X_b = len(X)
    U_b = len(U)  # 应该是得到未标签的batch_size数才对

    # step 1: Augmentation
    # 随机在输入上增加扰动
    X_cap = [(augmentation(x[0], aug_factor), x[1]) for x in X]  # shape unchanged
    #已標記數據 一次 weak augmentation


    # # U_cap = [[augmentation(u, aug_factor) for i in range(K)] for u in U] #U_cap is a list (length b) of list (length K)

    # U_cap = torch.from_numpy(U)  # [b, 1, D, H, W]
    # # 對爲標記數據平均k次 weak augmentation
    # # if GPU:
    # U_cap  = U_cap.cuda()
    
    # batchsize翻K倍
    U_cap = U.repeat(K, 1, 1, 1, 1)  # [K*b, 1, D, H, W]

    # 一样加上一些随机扰动
    U_cap += torch.clamp(torch.randn_like(U_cap) * 0.1, -0.2, 0.2)  # augmented.



    # step 2: label guessing
    with torch.no_grad():
        Y_u_tanh, Y_u = eval_net(U_cap)  # (K, n_class, 112, 112, 80), (K, n_class, 112, 112, 80) 已经是one_hot的形式了
        Y_u = F.softmax(Y_u, dim=1)  
    #print(Y_u.shape)

    guessed = torch.zeros(U.shape).repeat(1, 2, 1, 1, 1)  # empty label [b, 2, D, H, W]  这边是做成one-hot形式了

    #if GPU:
    guessed = guessed.cuda()
    for i in range(K):
        guessed += Y_u[i * U_b:(i + 1) * U_b]
    guessed /= K  # 将多次重复算个平均结果  原文的公式(6)

    # 此外在做一些处理, 得到伪标签
    #guessed = guessed ** (1 / T)  # 原文的公式(7)
    #guessed = guessed / guessed.sum(dim=1, keepdim=True)  # 原文的公式(7)
    pse_dis1 = guessed ** (1 / 0.2)
    pse_dis2 = (1 - guessed) ** (1 / 0.2)
    guessed = pse_dis1 / (pse_dis1 + pse_dis2)
    guessed = guessed.repeat(K, 1, 1, 1, 1)
    guessed = guessed.detach().cpu().numpy()  # shape [U_b * K,2,D,H,W]  得到伪标签
    pseudo_label = guessed
    U_cap = U_cap.detach().cpu().numpy()

    U_cap = list(zip(U_cap, guessed))  # 将伪标签和绕动数据合并


    ## Now we have X_cap ,list of (data, label) of length b, U_cap, list of (data, guessed_label) of length k*b

    # step 3: MixUp
    # original paper mathod

    x_mixup_mode, u_mixup_mode = mixup_mode[0], mixup_mode[1]

    W = X_cap + U_cap  # length = X_b + U_b * k, 将带标签的数据和伪标签数据合并
    random.shuffle(W)  # 随机打乱顺序
   
    if x_mixup_mode == 'w':
        X_prime = [mix_up(X_cap[i], W[i], alpha) for i in range(X_b)]  # 只取带标签的batch_size数进行融合
    elif x_mixup_mode == 'x':
        idxs = np.random.permutation(range(X_b))  # 保证取的范围在带标签的batch_size数的范围内, 取带标签的batch_size数个进行融合
        X_prime = [mix_up(X_cap[i], X_cap[idxs[i]], alpha) for i in range(X_b)]
    elif x_mixup_mode == 'u':
        idxs = np.random.permutation(range(U_b * K))[:X_b]  # 保证取的范围在不标签的batch_size数的范围内, 取带标签的batch_size数个进行融合
        X_prime = [mix_up(X_cap[i], U_cap[idxs[i]], alpha) for i in range(X_b)]
    elif x_mixup_mode == '_':
        X_prime = X_cap
    else:
        raise ValueError('wrong mixup_mode')

    if u_mixup_mode == 'w':  # 扣除前X_b个, 剩下进行融合
        U_prime = [mix_up(U_cap[i], W[X_b + i], alpha) for i in range(U_b * K)]
    elif u_mixup_mode == 'x':  # 保证取的范围在带标签的batch_size数的范围内, 取不带标签的batch_size数个进行融合
        idxs = np.random.permutation(range(U_b * K)) % X_b
        U_prime = [mix_up(U_cap[i], X_cap[idxs[i]], alpha) for i in range(U_b * K)]
    elif u_mixup_mode == 'u':  # 保证取的范围在不带标签的batch_size数的范围内, 取不带标签的batch_size数个进行融合
        idxs = np.random.permutation(range(U_b * K))
        U_prime = [mix_up(U_cap[i], U_cap[idxs[i]], alpha) for i in range(U_b * K)]  # 有問題???
    elif u_mixup_mode == '_':
        U_prime = U_cap
    else:
        raise ValueError('wrong mixup_mode')
    # print(U_cap[0][0].shape, U_cap[0][1].shape, X_cap[0][0].shape, X_cap[0][1].shape)
    # if DEBUG:
    # save_as_image(np.array([x[0] for x in U_prime]), f"../debug_output/u_prime_data")
    # save_as_image(np.array([x[1][[1], :, :, :] for x in U_prime]), f"../debug_output/u_prime_label")
    return X_prime, U_prime, pseudo_label


def mix_up(s1, s2, alpha):
    # print('??????', s1[0].shape, s1[1].shape, s2[0].shape, s2[1].shape)
    # s1, s2 are tuples(data, label)
    l = np.random.beta(alpha, alpha)  # 原文公式(8)
    l = max(l, 1 - l)  # 原文公式(9)

    x1, p1 = s1 
    x2, p2 = s2

    x = l * x1 + (1 - l) * x2  # 原文公式(10)
    p = l * p1 + (1 - l) * p2  # 原文公式(11)

    return (x, p)
