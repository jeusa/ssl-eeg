"""Perform metric calculation for supervised and self-supervised encoders: triplet accuracy, MAE.
Load files for loss and triplet accuracies. Perform UMAP on encoder output. Get PSD for EEG data."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import scipy
import os
from sklearn.metrics import mean_absolute_error
from umap import UMAP

from . import model, prediction as prd
from .train_sl import losses_path
from .preprocessing import s_freq

accuracies_path = os.path.join(model.models_path, "accuracies")

def load_loss(model_name, verbose=False):
    losses = None
    if verbose:
        print("Loading loss of", model_name)

    with open(os.path.join(losses_path, model_name + ".json"), "r") as lf:
        losses = json.load(lf)

    return losses


def load_loss_by_id(conf_id, val_i, head=False):
    loss = None

    if head:
        models_doc = pd.read_csv(model.models_doc_head_path, index_col=0)
        models_conf = pd.read_csv(model.models_conf_head_path, index_col=0)
    else:
        models_conf = pd.read_csv(model.models_conf_path, index_col=0)
        models_doc = pd.read_csv(model.models_doc_path, index_col=0)

    cur_conf = models_conf.loc[conf_id]
    cur_model = models_doc.loc[(models_doc["conf_id"]==conf_id) & (models_doc["val_idx"]==val_i)].iloc[0]
    model_name = cur_model["model_name"]

    loss = load_loss(model_name, True)

    if cur_conf["epochs"] > cur_model["epochs"]:
        print("Prepending loss history of base model")
        base_id = model.get_same_config(conf_id)
        base_loss = load_loss_by_id(base_id, val_i)

        for l in loss:
            loss[l] = base_loss[l] + loss[l]

    return loss


def load_margin_accuracies(model_name, set="validation"):
    supervision = "supervised"
    if "ssl" in model_name:
        supervision = "self-supervised"

    if supervision == "supervised":
        if set == "validation":
            file_name = model_name + "_nback"
        elif set == "test":
            file_name = model_name + "_nback_test"
            
    elif supervision == "self-supervised":
        if set == "validation":
            file_name = model_name
        elif set == "test":
            file_name = model_name + "_nback"
        elif set == "disjoint_test":
            file_name = model_name + "_disjoint_nback"

    with open(os.path.join(accuracies_path, file_name + ".json"), "r") as af:
        acc = json.load(af)
    
    return acc


def get_epochs_mean_losses(loss_his, epochs, last_batches=None):
    epoch_losses = []
    steps = int(len(loss_his) / epochs)

    for e in range(epochs):
        b = loss_his[steps*e : steps*(e+1)]
        mean_loss = sum(b) / steps

        if type(last_batches) is int:
            b = b[-last_batches:]
            mean_loss = sum(b) / last_batches
        elif type(last_batches) is float:
            b = b[int(-steps * last_batches):]
            mean_loss = sum(b) / (steps*last_batches)

        epoch_losses.append(mean_loss)

    return epoch_losses


def calc_triplet_accuracies(triplet_features, loss_margin, file_name=None, margins=[0,1,10,25,50,100,150,200,250,300]):
    marg_acc = {
        "margins": margins,
        "accuracies": []
    }
    
    for m in marg_acc["margins"]:
        acc = prd.get_triplet_accuracy(triplet_features, m)
        marg_acc["accuracies"].append(acc)
    
    if file_name:
        with open(os.path.join(accuracies_path, file_name + ".json"), "w") as af:
             json.dump(marg_acc, af)
    
    val_acc = prd.get_triplet_accuracy(triplet_features, loss_margin)

    return val_acc


def calc_triplet_accuracies_sl(model, triplets, chunks_data, loss_margin, file_name=None, margins=[0,1,10,25,50,100,150,200,250,300]):
    trp_loader = torch.utils.data.DataLoader(triplets, batch_size=256, shuffle=False)
    out = prd.get_triplet_loader_output_sl(model, trp_loader, chunks_data)
    val_acc = calc_triplet_accuracies(out, loss_margin, file_name, margins)

    return val_acc


def calc_triplet_accuracies_ssl(model, chunks_t, blocks_t, augmentation_scale, loss_margin, file_name=None, margins=[0,1,10,25,50,100,150,200,250,300]):  
    out = prd.generate_triplet_output_ssl(model, chunks_t, blocks_t, augmentation_scale=augmentation_scale)
    val_acc = calc_triplet_accuracies(out, loss_margin, file_name=file_name, margins=margins)

    return val_acc


def calc_mae(pred_df):
    pred_n = pred_df["pred_n"].to_numpy()
    true_n = pred_df["n"].to_numpy()
    return mean_absolute_error(true_n, pred_n)


def dim_red_umap(data, neighbors=60, min_dist=1.):
    umap = UMAP(n_neighbors=neighbors, n_components=2, metric="euclidean", min_dist=min_dist)
    umap_results = umap.fit_transform(data)
    return umap_results


def get_psds(ses):
    psd_ses = []

    for s in ses:
        eeg_data = s[:8,:]
        freqs, psd = scipy.signal.welch(eeg_data, fs=s_freq)
        psd_ses.append(psd)

    psd_ses = np.array(psd_ses)
    return freqs, psd_ses