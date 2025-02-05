"""General preprocessing of EEG data. Normalization, band-pass filtering,
splitting in training and validation sets."""

import pandas as pd
import mne
import torch
from sklearn.preprocessing import RobustScaler


s_freq = 250 # sampling frequency in Hz
l_ffreq = 0.1 # high-pass filter frequency in Hz
h_ffreq = 45 # low-pass filter frequency in Hz
s_time = 2 # length of one sample in seconds
s_length = s_freq * s_time


def get_folds(data, k_folds):
    folds = []

    for i in range(k_folds):
        if type(data) is torch.Tensor:
            f = data[i::k_folds].clone() 
        else:
            f = data[i::k_folds].copy()
        
        folds.append(f)
    
    return folds


def get_train_val_sets(folds, val_fold_idx, batch_size=256, val_batch_size=2048, return_dataloader=True):
    
    if type(folds[0]) is pd.DataFrame:
        val_set = folds[val_fold_idx].copy()
    elif type(folds[0]) is torch.Tensor:
        val_set = folds[val_fold_idx].clone()
    
    train_set = folds.copy()
    train_set.pop(val_fold_idx)
    
    if type(val_set) is pd.DataFrame:
        train_set = pd.concat(train_set)
        
        return train_set.sort_index(), val_set
                
    elif type(val_set) is torch.Tensor:
        
        train_set = torch.cat(train_set, dim=0)

        if return_dataloader:
            trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
            valloader = torch.utils.data.DataLoader(val_set, batch_size=val_batch_size, shuffle=True)

            return trainloader, valloader
        else:
            return train_set, val_set
    

def normalize_data(chunks_data):
    data = chunks_data.clone()

    for i, s in enumerate(data):
        data[i] = normalize_chunk(s)
    
    return data

def normalize_chunk(chunk_data):
    data = chunk_data.clone()

    for i, c in enumerate(data):
        min_v = torch.min(c)
        max_v = torch.max(c)

        data[i] = (c-min_v) / (max_v-min_v)
    
    return data


def normalize_triplet_data(data):
    d = data.clone()

    for t in d:
        for s in t:
            for i, c in enumerate(s):
                min_v = torch.min(c)
                max_v = torch.max(c)

                s[i] = (c-min_v) / (max_v-min_v)

    return d


def normalize_data_robust(data):
    d = data.clone()

    for i, s in enumerate(d):
        s_norm = RobustScaler().fit_transform(s.T).T
        d[i] = torch.from_numpy(s_norm)
    
    return d


def normalize_triplet_data_robust(data):
    d = data.clone()

    for t in d:
        for i, s in enumerate(t):
            s_norm = RobustScaler().fit_transform(s.T).T
            t[i] = torch.from_numpy(s_norm)
    
    return d


def make_mne_raw(eeg, verbose=True):
    info = mne.create_info(8, s_freq, ["eeg"]*8)
    raw = mne.io.RawArray(eeg, info, verbose=verbose)

    return raw


def filter_eeg(eeg, l_freq=None, h_freq=None, verbose=True, trans_band_auto=False):
    l_trans_bandwidth = 0.05
    h_trans_bandwidth = 2
    if trans_band_auto:
        l_trans_bandwidth, h_trans_bandwidth = "auto", "auto"
    
    if l_freq == None:
        l_freq = l_ffreq

    if h_freq == None:
        h_freq = h_ffreq
    elif h_freq == 0:
        h_freq = None

    raw = make_mne_raw(eeg.copy(), verbose)
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=verbose, l_trans_bandwidth=l_trans_bandwidth, h_trans_bandwidth=h_trans_bandwidth)

    return raw["data"][0]