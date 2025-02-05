"""Get feature representations from encoders. Do knn classification. """

import pandas as pd
import torch
import sys

from tqdm.auto import trange
from torch import nn

from . import preprocessing as pr, preprocessing_nback as prn, preprocessing_unlabeled as pru


def get_model_output(model, data, device=None):
    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)
    data = data.to(device)
    model.eval()

    with torch.no_grad():
        out = model.forward_once(data)
    
    return out


def get_model_output_loader(model, data_loader, device=None):
    it = iter(data_loader)
    out_batches = []

    for b in trange(len(data_loader), desc="Prediction"):
        batch = next(it)
        out = get_model_output(model, batch, device)
        out_batches.append(out)
    
    data_out = torch.concat(out_batches)
    
    return data_out


def get_model_output_triplet(model, data, device=None):
    a = get_model_output(model, data[:,0,:,:], device)
    p = get_model_output(model, data[:,1,:,:], device)
    n = get_model_output(model, data[:,2,:,:], device)

    return torch.stack((a,p,n), dim=1)


# labeled triplets
def get_triplet_loader_output_sl(model, data_loader, chunks_data, device=None):
    trp_it = iter(data_loader)
    out_batches = []

    for b in trange(len(data_loader), desc="Prediction"):
        batch_trp = next(trp_it)
        batch = prn.generate_data_from_triplets(batch_trp, chunks_data)
        out = get_model_output_triplet(model, batch, device=device)
        out_batches.append(out)

    data_out = torch.concat(out_batches)

    return data_out


# triplets with augmented positive samples
def get_triplet_loader_output_ssl(model, data_loader, chunks_t, blocks_t, device=None, augmentation_scale=1):
    trp_it = iter(data_loader)
    out_batches = []

    for b in trange(len(data_loader), desc="Prediction"):
        batch_trp = next(trp_it)
        batch = pru.generate_triplet_data(batch_trp, chunks_t, blocks_t, augmentation_scale)
        out = get_model_output_triplet(model, batch, device=device)
        out_batches.append(out)

    data_out = torch.concat(out_batches)

    return data_out

# triplets with augmented positive samples
def generate_triplet_output_ssl(model, chunks_t, blocks_t, augmentation_scale=1):
    prepared_triplets = pru.prepare_triplets(chunks_t)
    trp_loader = torch.utils.data.DataLoader(prepared_triplets, batch_size=256, shuffle=False)
    data_out = get_triplet_loader_output_ssl(model, trp_loader, chunks_t, blocks_t, augmentation_scale=augmentation_scale)

    return data_out


def calc_triplet_loss(triplet_features, loss_margin, distance_type="euclidean", reduction="mean"):
    # distance function
    dis_f = None

    if distance_type == "euclidean":
        dis_f = nn.PairwiseDistance(p=2)
    elif distance_type == "cosine":
        dis_f = lambda x, y: 1.0 - torch.nn.functional.cosine_similarity(x, y, dim=0)

    criterion = nn.TripletMarginWithDistanceLoss(distance_function=dis_f, margin=loss_margin, reduction=reduction)

    return criterion(triplet_features[:,0,:], triplet_features[:,1,:], triplet_features[:,2,:])


def get_triplet_accuracy(triplet_features, loss_margin, distance_type="euclidean"):
    loss = calc_triplet_loss(triplet_features, loss_margin, distance_type, reduction="none")
    acc = len(loss[loss == 0]) / len(loss)

    return acc

# for self supervised models
def calc_triplet_accuracy_ssl(model, chunks_t, blocks_t, loss_margin):
    data_out = generate_triplet_output_ssl(model, chunks_t, blocks_t)
    
    return get_triplet_accuracy(data_out, loss_margin)


def get_model_output_n(model, chunks_df, blocks_df, as_df=True, device=None):
    X, Y = prn.get_samples_data(chunks_df, blocks_df)
    X = pr.normalize_data(X)
    X_out = get_model_output(model, X, device)

    if as_df:
        out_df = pd.DataFrame({
            "n": [int(a) for a in Y],
            "output": [a for a in X_out.cpu()]
        })

        return out_df
    
    else:
        return X_out, Y


def make_predictions(chunks, neigh_chunks, model, k, blocks, distance_type="euclidean", pool_labels=False, device=None):
    train_out = get_model_output_n(model, neigh_chunks, blocks, device=device)
    test_out = get_model_output_n(model, chunks, blocks, device=device)

    pred_out = test_out.copy()
    pred_out["pred_n"] = -1

    for i, row in pred_out.iterrows():
        pred_n = make_single_prediction(row["output"], train_out, k, distance_type)
        pred_out.loc[i, "pred_n"] = pred_n
    
    if pool_labels:
        pred_out.loc[pred_out["n"]==0, "n"] = 1
        pred_out.loc[pred_out["n"]==3, "n"] = 2
        pred_out.loc[pred_out["pred_n"]==0, "pred_n"] = 1
        pred_out.loc[pred_out["pred_n"]==3, "pred_n"] = 2

    pred_out["pred_correct"] = (pred_out["n"] == pred_out["pred_n"])
    pred_count = pred_out.groupby("pred_correct").count()

    acc = pred_count.loc[True]["n"] / pred_out.shape[0]

    return pred_out, acc


def make_single_prediction(sample_out, neigh_df, k, distance_type="euclidean"):
    closest = get_closest_neighbors(sample_out, neigh_df, k, distance_type)
    pred, max_pred, pred_n = get_label_predictions(closest)

    return pred_n


def get_closest_neighbors(sample_out, neigh_df, k, distance_type="euclidean"):
    n_df = neigh_df.copy()
    t_out = sample_out.clone()

    out_neigh = torch.stack(tuple(n_df["output"]))
    
    dis_f = None
    if distance_type == "euclidean":
        dis_f = torch.nn.PairwiseDistance(p=2)
    elif distance_type == "cosine":
        dis_f = lambda x, y: 1.0 - torch.nn.functional.cosine_similarity(x, y, dim=1)

    if t_out.dim()==1:
        t_out = torch.reshape(t_out, (1, t_out.shape[0]))
    dist = dis_f(out_neigh, t_out)

    n_df.insert(2, "dist", dist)
    n_df = n_df.sort_values("dist")

    return n_df.iloc[:k]


def get_label_predictions(close_df):
    k = close_df.shape[0]
    pred = []
    count_df = close_df.groupby("n").count()

    # calc probability for classes according to the k closest neighbors
    for i in range(4):
        p = 0
        if i in count_df.index:
            p = count_df.loc[i]["output"] / k

        pred.append(p)

    max_prob = max(pred)
    max_pred = []

    # determine most probable class(es)
    for i in range(4):
        if pred[i] == max_prob:
            max_pred.append(i)

    # if two or more classes have highest probability, determine for which class the sum of
    # the distances is less
    n_smallest_dist = max_pred[0]
    if len(max_pred) > 1:
        smallest_dist = sys.float_info.max

        for n in max_pred:
            dist = sum(close_df.loc[close_df["n"]==n]["dist"])
            if (dist < smallest_dist):
                smallest_dist = dist
                n_smallest_dist = n

    return pred, max_pred, n_smallest_dist