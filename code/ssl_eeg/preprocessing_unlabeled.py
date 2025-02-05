"""Preprocessing of unlabeled EEG data. Load CSV files and make data frame.
Generate triplets. Options to filter and normalize the data."""

import numpy as np
import pandas as pd
import torch
import platform

from . import util, preprocessing as pr, augmentation as aug

data_path = "/home/jeusa/Files/Documents/Medieninformatik/Bachelorarbeit/Daten/unlabeled/"
if platform.system() == "Windows":
    data_path = "h:\Documents\Medieninformatik\Bachelorarbeit\Daten\\unlabeled"


def arange_data(verbose=False, lowpass=None, trans_band_auto=False, sessions=None):
    blocks = make_blocks_df()
    if not sessions == None:
        blocks = blocks.loc[blocks["session_no"].isin(sessions)]
    blocks = filter_blocks_df(blocks, verbose=verbose, lowpass=lowpass, trans_band_auto=trans_band_auto)
    chunks_1 = make_chunks_df(blocks)
    chunks_2 = make_chunks_df(blocks, offset=pr.s_time//2)
    chunks = pd.concat([chunks_1, chunks_2]).sort_values("start_idx").reset_index(drop=True)

    return blocks, chunks


# removing rows with Validation Indicator == 0 and forming blocks_df
def make_blocks_df():
    blocks = pd.DataFrame()
    data_dfs = []

    for f in util.list_files(data_path, sort_by="date"):
        d = pd.read_csv(f)
        data_dfs.append(d)

    for i in range(len(data_dfs)):
        df = data_dfs[i].copy()
        df["session_no"] = i+1
        df["block_no"] = 1
        df_bad = df.loc[df["Validation Indicator"]==0]

        if df_bad.shape[0] > 0:
            df1 = df.loc[:df_bad.iloc[0].name-1].copy()
            df2 = df.loc[df_bad.iloc[-1].name+1:].copy()
            df2["block_no"] = 2
            blocks = pd.concat([blocks, df1, df2])
        else:
            blocks = pd.concat([blocks, df])

    blocks_df = blocks.reset_index(drop=True)
    blocks_df = blocks_df.drop(columns=["Accelerometer X", "Accelerometer Y", "Accelerometer Z", "Gyroscope X", "Gyroscope Y", "Gyroscope Z", "Battery Level", "Counter", "Validation Indicator"])
    
    return blocks_df


def make_chunks_df(blocks_df, offset=0):
    df = pd.DataFrame()
    offset = pr.s_freq * offset

    for i, b in blocks_df.groupby(["session_no", "block_no"]):
        rel_idx = b.iloc[offset::pr.s_length,:].index.values
        
        b_data = pd.DataFrame({
                "start_idx": rel_idx[:-1],
                "end_idx": rel_idx[1:],
                "session_no": i[0],
                "block_no": i[1]
            })
        
        b_data["end_idx"] -= 1

        df = pd.concat([df, b_data])

    df = df.reset_index(drop=True)

    return df


def get_samples_data_from_df(chunks_df, blocks_df):
    X = np.zeros([chunks_df.shape[0], 8, pr.s_length])

    for i in range(chunks_df.shape[0]):
        s = chunks_df.iloc[i]
        s_data = blocks_df.loc[s["start_idx"]:s["end_idx"]]
        X[i] = s_data.iloc[:, 0:8].to_numpy().T

    return torch.from_numpy(X).float()


def get_samples_data(chunks_t, blocks_t):
    blocks = blocks_t.clone()[:8,:]

    X = np.zeros([chunks_t.shape[0], 8, pr.s_length])

    for i, c in enumerate(chunks_t):
        X[i] = blocks[:, c[0]:c[1]+1]
    
    return torch.from_numpy(X).float()


def get_random_negatives(size):
    negs = np.zeros(shape=(size, 1), dtype=int)
    idx_list = np.arange(size)
    rand_del_range = 6 # equals at least plus-minus 6s

    for i, c in enumerate(negs):
        cur_idx_list = np.delete(idx_list, slice(max(0, i-rand_del_range), i+rand_del_range+1)) # remove adjacent chunks
        negs[i] = np.random.choice(cur_idx_list)
    
    return negs


def get_random_negatives_same_session(chunks):
    negs = np.zeros(shape=(chunks.shape[0], 1), dtype=int)

    for ses_no in np.unique(chunks[:,2]):

        filter_ses = chunks[:,2] == ses_no # filters all chunks from the current session
        first_idx = np.argmax(filter_ses).item() # start index of session
        end_idx = chunks.shape[0] - np.argmax(np.flip(torch.Tensor.numpy(filter_ses))) # start index of next session

        idx_list = np.arange(first_idx, end_idx) # indexes of current session
        rand_del_range = 6 # equals at least plus-minus 6s

        for i, c in enumerate(idx_list):
            cur_idx_list = np.delete(idx_list, slice(max(0, i-rand_del_range), i+rand_del_range+1)) # remove adjacent chunks
            negs[c] = np.random.choice(cur_idx_list)
    
    return negs


def get_random_augmentations(size, as_int=True):
    augm = np.zeros(shape=(size, 2), dtype=object)
    augmentations = np.array(["time_shift", "masking", "filter", "noise", "frequency_shift"])

    if as_int:
        augmentations = np.arange(augmentations.shape[0])

    for i, a in enumerate(augm):
        augm[i] = np.random.choice(augmentations, 2, replace=False)

    if as_int:
        augm = augm.astype("int32")
        return torch.from_numpy(augm)
    
    return augm


def prepare_triplets(chunks): 
    anch = np.arange((chunks.shape[0]))
    anch = np.reshape(anch, (anch.shape[0], 1)) # indexes of anchors and positive chunks
    negs = get_random_negatives_same_session(chunks) # indexes of random negative chunks
    augm = get_random_augmentations(chunks.shape[0]) # list of random augmentations as int, 2 augmentations per positive chunks
    
    return torch.from_numpy(np.concatenate([anch, negs, augm], axis=1))


def generate_triplet_data(prepared_triplets, chunks_t, blocks_t, scale=1):
    pt = prepared_triplets.clone().type(torch.LongTensor)

    anc = chunks_t[pt[:,0]] # anchor chunks
    anc_data = get_samples_data(anc, blocks_t) # anchors eeg data
    anc_data = pr.normalize_data(anc_data) # normalize anchors

    pos = anc.clone() # positive chunks

    neg = chunks_t[pt[:,1]]
    neg_data = get_samples_data(neg, blocks_t) # negatives eeg data
    neg_data = pr.normalize_data(neg_data) # normalize negatives

    # order of augmentations and preprocessing:
    # time shift
    # frequency shift
    # band stop filter
    # normalization
    # noise
    # masking

    # apply time shift if applicable
    pos_ts = pos.clone() # time shifted positive chunks
    for i, p in enumerate(pos_ts):
        cur_aug = pt[i,2:]

        if 0 in cur_aug: # apply time shift
            pos_ts[i] = aug.time_shift(p, blocks_t, scale=scale)
    
    pos_data = get_samples_data(pos_ts, blocks_t) # positives eeg data

    # apply frequency shift and band-stop filter if applicable
    for i, p in enumerate(pos_data):
        cur_aug = pt[i, 2:]

        if 4 in cur_aug: # apply frequency shift
            pos_data[i] = aug.frequency_shift(p, scale=scale)

        if 1 in cur_aug: # apply filter
            pos_data[i] = aug.band_stop_filter(p, scale=scale)
    
    pos_data_norm = pr.normalize_data(pos_data) # normalize positives

    # add noise and masking if applicable
    for i, p in enumerate(pos_data_norm):
        cur_aug = pt[i, 2:]

        if 2 in cur_aug: # add noise
            pos_data_norm[i] = aug.add_noise(p, scale=scale)
        
        if 3 in cur_aug: # masking
            pos_data_norm[i] = aug.mask(p, scale=scale)
    
    return torch.stack([anc_data, pos_data_norm, neg_data], axis=1)


def filter_blocks_df(blocks_df, lowpass=None, trans_band_auto=False, verbose=False):
    b = pd.DataFrame()

    for g, f in blocks_df.groupby(["session_no", "block_no"]):
        frame = f.copy()
        
        if g[1] == 1:
            frame = frame.iloc[pr.s_freq * 30:] # remove first 30s of every session
            
        cur_eeg = frame.to_numpy()[:,:8].T
        fil_eeg = pr.filter_eeg(cur_eeg, h_freq=lowpass, trans_band_auto=trans_band_auto, verbose=verbose)

        f = pd.DataFrame(fil_eeg.T, columns=blocks_df.columns[:8])
        f["session_no"] = g[0]
        f["block_no"] = g[1]

        b = pd.concat([b, f])

    b = b.reset_index(drop=True)
    
    return b