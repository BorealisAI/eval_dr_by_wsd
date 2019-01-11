# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

import ot
import faiss


def get_self_knn_idx(data, k):
    dim = data.shape[1]
    data = np.ascontiguousarray(data.astype('float32'))
    try:
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = 1
        ngpu = 2
        res = [faiss.StandardGpuResources() for i in range(ngpu)]
        index = faiss.GpuIndexFlatL2(res[1], dim, cfg)
    except:
        index = faiss.IndexFlatL2(dim)
    index.add(data)

    _, kidx = index.search(data, k + 1)

    return kidx[:, 1:]


def get_mean_pdist(x, num=None):
    if num is None:
        num = len(x)
    rand_idx = random.sample(range(len(x)), min(len(x), num))
    distmat = squareform(pdist(x[rand_idx]))
    return distmat[np.triu_indices(distmat.shape[0])].mean()


def normalize_data_by_mean_pdist(x, num=None):
    if num is None:
        num = len(x)
    return x / get_mean_pdist(x, num)


def get_emd2(pts1, pts2):
    distmat = ot.dist(pts1, pts2)
    a = ot.unif(len(pts1))
    b = ot.unif(len(pts2))
    return ot.emd2(a, b, distmat)


def get_wsd_scores(x, y, k, num_meandist=None):
    kidx_x = get_self_knn_idx(x, k)
    kidx_y = get_self_knn_idx(y, k)
    x = normalize_data_by_mean_pdist(x, num_meandist)
    y = normalize_data_by_mean_pdist(y, num_meandist)

    assert len(kidx_x) == len(kidx_x) == len(x) == len(y)

    discontiuity = np.array(
        [get_emd2(y[kidx_x[i]], y[kidx_y[i]]) for i in range(len(x))]
    )
    manytoone = np.array(
        [get_emd2(x[kidx_x[i]], x[kidx_y[i]]) for i in range(len(x))]
    )

    return discontiuity, manytoone
