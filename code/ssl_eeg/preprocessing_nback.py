"""Preprocessing of n-back EEG data. Load XDF files and make data frame.
Generate triplets. Options to filter and normalize the data."""

import pandas as pd
import numpy as np
import pyxdf
import re
import torch
import os

from . import util, preprocessing as pr


root_dir = os.getcwd().split("code")[0]
data_dir = os.path.join(root_dir, "data", "n-back")
ses_files = ["n-back-session001.xdf", "n-back-session002.xdf", "n-back-session003.xdf", "n-back-session004.xdf"]
ses_files = [os.path.join(data_dir, f) for f in ses_files]
blocks_path = os.path.join(data_dir, "blocks_data.csv")


def arange_data(filter_data=True, lowpass=None, trans_band_auto=False, verbose=False, sessions=None):
    blocks = extract_blocks(verbose=verbose, filter_data=filter_data, lowpass=lowpass, trans_band_auto=trans_band_auto)
    if not sessions == None:
        blocks = blocks.loc[blocks["session_no"].isin(sessions)]
    chunks_1 = make_chunks_df(blocks)
    chunks_2 = make_chunks_df(blocks, offset=pr.s_time//2)
    chunks = pd.concat([chunks_1, chunks_2]).sort_values("start_idx").reset_index(drop=True)

    return blocks, chunks


def get_train_test_sets(chunks, test_session=None):
    train_chunks = chunks.loc[chunks["session_no"] != test_session].reset_index(drop=True)
    test_chunks = chunks.loc[chunks["session_no"] == test_session].reset_index(drop=True)

    return train_chunks, test_chunks


def generate_data_from_triplets(triplets_tensor, chunks_data, normalize=False):
    anchors, positives, negatives, n_pos, n_neg = get_triplets_data(triplets_tensor, chunks_data)
    data = torch.stack((anchors, positives, negatives), dim=1)

    if normalize:
        data = pr.normalize_triplet_data(data)
    
    return data


def get_triplets_data(triplets_tensor, chunks_data):
    trp = triplets_tensor.T.clone()
    
    anchors_data = chunks_data[trp[0]]
    positives_data = chunks_data[trp[1]]
    negatives_data = chunks_data[trp[2]]
    pos_ns = trp[3]
    neg_ns = trp[4]
    
    return anchors_data, positives_data, negatives_data, pos_ns, neg_ns


def make_triplets(chunks_df):
    ci_n = np.stack([chunks_df.index, chunks_df["n"]], axis=1)
    pos_pairs = get_positive_pairs((ci_n))
    trp = get_random_negatives(ci_n, pos_pairs)

    return trp


def get_positive_pairs(chunks_n_array):
    ci_n = chunks_n_array.copy()
    pairs = []

    for i, (c, n) in enumerate(ci_n):
        other_c = np.delete(ci_n, i, axis=0)
        same_n = other_c[other_c[:,1] == n]

        anchors = np.repeat([c], same_n.shape[0])
        pos_pairs = np.stack([anchors, same_n[:,0], np.repeat([n], same_n.shape[0])], axis=1)
        pairs.append(pos_pairs)
        
    pairs = np.concatenate(pairs)

    return pairs # [[anchor, positive, n]]


def get_random_negatives(chunks_n_array, pos_pairs_array):
    ci_n = chunks_n_array.copy()
    rng = np.random.default_rng()
    trp = []

    for n in range(4):
        cur_n_pairs = pos_pairs_array[pos_pairs_array[:,2] == n]
        other_n = ci_n[ci_n[:,1] != n]
        negs_idx = rng.choice(other_n.shape[0], cur_n_pairs.shape[0]) # choose random indices for negatives with other n value
        negs = other_n[negs_idx]
        trp.append(np.c_[cur_n_pairs[:,:2], negs[:,0], cur_n_pairs[:,2], negs[:,1]])
        
    trp = np.concatenate(trp)

    return trp # [[anchor, positive, negative, pos_n, neg_n]]


def get_samples_data(chunks_df, blocks_df):

    X = np.zeros([chunks_df.shape[0], 8, pr.s_length])
    Y = np.zeros(chunks_df.shape[0])

    for i in range(chunks_df.shape[0]):

        s = chunks_df.iloc[i]
        s_data = blocks_df.loc[s["start_idx"]:s["end_idx"]]
        Y[i] = int(s_data.iloc[0]["n"])
        X[i] = s_data.iloc[:, 1:9].to_numpy().T

    return torch.from_numpy(X).float(), torch.from_numpy(Y).float()


def train_test_split(chunks_df, split):
    train = pd.DataFrame()
    test = pd.DataFrame()

    for n_label, frame in chunks_df.sample(frac=1).groupby(["session_no", "n"]):

        split_idx = int(np.round(frame.shape[0] * split))
        train = pd.concat([train, frame.iloc[:split_idx]])
        test = pd.concat([test, frame.iloc[split_idx:]])

    train = train.sample(frac=1)
    test = test.sample(frac=1)

    return train.sort_index(), test.sort_index()


# offset in seconds
def make_chunks_df(blocks_df, offset=0):
    df = pd.DataFrame()
    offset = pr.s_freq * offset

    for n_label in range(4):
        for i, b in blocks_df.loc[blocks_df["n"]==n_label].groupby(["session_no", "block_no"]):
            rel_idx = b.iloc[offset::pr.s_length,:].index.values

            b_data = pd.DataFrame({
                "start_idx": rel_idx[:-1],
                "end_idx": rel_idx[1:],
                "n": n_label,
                "session_no": i[0],
                "block_no": i[1],
                "offset": offset
            })
            b_data["end_idx"] -= 1

            df = pd.concat([df, b_data])

    df = df.reset_index(drop=True)

    return df


def extract_blocks(save=False, verbose=True, filter_data=True, lowpass=None, trans_band_auto=False):
    blocks_df = pd.DataFrame()

    for i, f in enumerate(ses_files):
        eeg, markers = read_xdf(f)
        if filter_data:
            eeg["time_series"] = pr.filter_eeg(eeg["time_series"], h_freq=lowpass, trans_band_auto=trans_band_auto, verbose=verbose)
        eeg, markers = make_session_dfs(eeg, markers)
        ses = combine_dfs(eeg, markers, session_no=i+1)
        bl = make_blocks_df(ses)
        blocks_df = pd.concat([blocks_df, bl])

    if save:
        blocks_df.to_csv(blocks_path, index=False)

    return blocks_df.reset_index(drop=True)


def read_xdf(xdf_file_path):

    data, header = pyxdf.load_xdf(xdf_file_path)
    unicorn, n_back = None, None

    for t in data:
        if t["info"]["name"][0] =="n-back":
            n_back = t
        else:
            unicorn = t

    markers = {}
    markers["time_series"]  = util.flatten(n_back["time_series"])
    markers["time_stamps"] = n_back["time_stamps"]

    eeg = {}
    # eeg["time_series"] = unicorn["time_series"].T[:8] / (10**6) # micro Volt to Volt
    eeg["time_series"] = unicorn["time_series"].T[:8]
    eeg["time_stamps"] = unicorn["time_stamps"]

    return eeg, markers


def make_session_dfs(eeg, n_back):

    markers_df = pd.DataFrame({
    "marker": n_back["time_series"],
    "time_stamp": n_back["time_stamps"]
    })

    eeg_df = pd.DataFrame({
        "time_stamp": eeg["time_stamps"],
        "eeg_1": eeg["time_series"][0],
        "eeg_2": eeg["time_series"][1],
        "eeg_3": eeg["time_series"][2],
        "eeg_4": eeg["time_series"][3],
        "eeg_5": eeg["time_series"][4],
        "eeg_6": eeg["time_series"][5],
        "eeg_7": eeg["time_series"][6],
        "eeg_8": eeg["time_series"][7],
    })

    return eeg_df, markers_df


def combine_dfs(eeg, markers, session_no=1):
    eeg_df = eeg.copy()
    markers_df = markers.copy()

    eeg_df["n_back_marker"] = ""
    eeg_df["n"] = -1
    eeg_df["session_no"] = session_no

    for i, mark in markers_df.iterrows():

        n = -1
        n_match = re.search("\d", mark["marker"])
        if not n_match == None:
            n = n_match.group()

        eeg_df.loc[eeg_df["time_stamp"] >= mark["time_stamp"], ["n_back_marker", "n"]] = [mark["marker"], n]

    eeg_df["n"] = eeg_df["n"].astype(int)

    return eeg_df


def make_blocks_df(ses):
    blocks_df = ses.copy()

    blocks_df = blocks_df.loc[blocks_df["n_back_marker"].str.startswith("n=")]
    blocks_df["block_no"] = blocks_df["n"].diff().ne(0).cumsum()
    blocks_df = blocks_df.drop(columns="n_back_marker")

    return blocks_df
