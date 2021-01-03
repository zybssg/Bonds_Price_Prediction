import numpy as np


def standardization(data, mu, sigma):
    if mu is None or sigma is None:
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
    std = (data - mu) / sigma

    return mu, sigma, std


def re_standardization(data, mu, sigma):
    re_std = (data * sigma) + mu

    return re_std


def get_feats_and_labels(data, seq_len):
    assert len(data.shape) == 2
    feats, labels = [], []
    idx = 0
    cur_bond_id = data[idx, 0]
    while idx+seq_len < len(data):
        if data[idx+seq_len, 0] == cur_bond_id:
            feats.append(data[idx:idx+seq_len, 1:])
            labels.append(data[idx+seq_len, -1])
            idx += 1
        else:
            idx += seq_len
            cur_bond_id = data[idx, 0]
    feats = np.array(feats)
    labels = np.array(labels)
    assert len(feats) == len(labels)
    data = {'feats':feats, 'labels':labels}
    
    return data
