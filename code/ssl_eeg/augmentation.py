"""Perform randomized augmentations on an EEG chunk. The five augmentations are:
time shift, masking, band-stop filtering, adding noise, shifting frequencies"""

import torch
import mne
import random
import numpy as np
from scipy.fft import rfft, rfftfreq, irfft

from . import preprocessing as pr


def time_shift(chunk, blocks_t, return_eeg=False, scale=1):
    shift_range = 60 * scale # equals ca. plus/minus 0.25 seconds if scale==1
    shift_min = 10

    rand = 0
    p = chunk.clone()
    valid_p = False

    while not valid_p:
        rand = int(random.random() * shift_range*2 - shift_range)

        if abs(rand) < shift_min:
            continue

        p[0] = chunk[0]+rand
        p[1] = chunk[1]+rand

        try:
            if p[0] < 0:
                continue
            elif (chunk[2:] != blocks_t[-2:, p[1]]).any(): # values of positive sample shift into next session/block
                continue
        except Exception as e:
            print("Positive chunk caused exception in time_shift:", p)
            print("Aborting time shift.")
            print(e)
            p = chunk.clone()
        
        valid_p = True
    
    if return_eeg:
        p_data = blocks_t[:8, p[0]:p[1]+1]
        return p, p_data
    else:
        return p


def mask(chunk_data, scale=1):
    masking_max = 180 * scale
    masking_min = 20

    rand_range = 0
    p_data = chunk_data.clone()

    for c in range(len(p_data)):
        rand_range = int(random.random() * (masking_max-masking_min) + masking_min)
        rand_start = int(random.random() * (pr.s_length-rand_range))

        p_data[c, rand_start:rand_start+rand_range] = 0.5

    return p_data


def band_stop_filter(chunk_data, verbose=False, scale=1):
    filter_width = round(5 * scale)
    filter_min = 3
    filter_max = 83 - filter_width
    rand_filter = random.random() * (filter_max-filter_min) + filter_min

    raw = pr.make_mne_raw(chunk_data.clone(), verbose)
    raw.filter(l_freq=rand_filter+filter_width, h_freq=rand_filter, l_trans_bandwidth=2, h_trans_bandwidth=2, filter_length=pr.s_length-1, verbose=verbose)

    return torch.from_numpy(raw["data"][0])


def add_noise(chunk_data, verbose=False, scale=1):
    std_min = 0.02
    std_max = 0.2 * scale
    rand_std = random.random() * (std_max-std_min) + std_min

    raw = pr.make_mne_raw(chunk_data.clone(), verbose)
    cov = mne.make_ad_hoc_cov(raw.info, std=rand_std, verbose=verbose)
    raw.set_eeg_reference("average", projection=True, verbose=verbose)
    mne.simulation.add_noise(raw, cov, verbose=verbose)

    return torch.from_numpy(raw["data"][0])


def frequency_shift(chunk_data, scale=1):
    if torch.is_tensor(chunk_data):
        chunk_n = torch.Tensor.numpy(chunk_data.clone())
    else:
        chunk_n = chunk_data.copy()
    
    ff = rfft(chunk_n) # fast fourier transformation
    freqs = rfftfreq(chunk_n.shape[1], 1/pr.s_freq)
    shift_freq = round(2 * scale) # hz
    shift_s = np.where(freqs == abs(shift_freq))[0][0] # hz to data points
    pad = np.repeat([0], shift_s)

    sff = ff.copy()[:,1:] # remove frequency == 0
    rand = random.random()
    for i, ch in enumerate(sff):
        if rand < 0.5: # shift left
            sff[i] = np.concatenate([ch, pad])[shift_s:]
        else: # shift right
            sff[i] = np.concatenate([pad, ch])[:-shift_s]
        
    return torch.from_numpy(irfft(sff, pr.s_freq * pr.s_time)) # inverse fast fourier transformation