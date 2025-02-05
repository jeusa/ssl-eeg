"""Train a self-supervised encoder."""

import os
import torch
import numpy as np
import json
from tqdm.auto import trange
from torch import nn
from codecarbon import EmissionsTracker
from copy import deepcopy

from . import preprocessing_unlabeled as pru, prediction as prd
from .model import models_path, save_model
from .train_sl import losses_path, device


def train(model, epochs, blocks, train_chunks, val_chunks, batch_size, val_batch_size, name, learning_rate=0.001, loss_margin=1, distance_type="euclidean", augmentation_scale=1, val_blocks_tensor=None):
    tracker = EmissionsTracker(project_name=name, output_dir=models_path, log_level="warning")
    tracker.start()

    if val_blocks_tensor == None:
        val_blocks_tensor = blocks
    
    # distance function
    dis_f = None
    if distance_type == "euclidean":
        dis_f = nn.PairwiseDistance(p=2)
    elif distance_type == "cosine":
        dis_f = lambda x, y: 1.0 - torch.nn.functional.cosine_similarity(x, y, dim=0)

    model.to(device)
    criterion = nn.TripletMarginWithDistanceLoss(distance_function=dis_f, margin=loss_margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    max_steps = int(np.ceil(train_chunks.shape[0] / batch_size))
    
    metrics = {
        "train_batch_loss": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    for epoch in trange(epochs, desc="Training"):
        # set new random negative samples and augmentations
        pt_train = pru.prepare_triplets(train_chunks)
        pt_val = pru.prepare_triplets(val_chunks)

        train_loader = torch.utils.data.DataLoader(pt_train, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(pt_val, batch_size=val_batch_size, shuffle=True)
        train_it = iter(train_loader)
        
        # training
        model.train()
        for b in trange(max_steps, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            
            batch_trp = next(train_it)
            batch = pru.generate_triplet_data(batch_trp, train_chunks, blocks, scale=augmentation_scale)
            batch = batch.to(device)
            
            net_out = model(batch[:, 0, :, :], batch[:, 1, :, :], batch[:, 2, :, :])

            batch_loss = criterion(net_out[0], net_out[1], net_out[2])
            batch_loss.backward()
            optimizer.step()

            metrics["train_batch_loss"].append(batch_loss.item())
        
        # calc loss on validation set after every epoch
        print("Calculating metrics...")
        val_loss, val_acc = get_metrics(model, val_loader, val_chunks, val_blocks_tensor, loss_margin, distance_type, augmentation_scale=augmentation_scale)
        train_loss, train_acc = get_metrics(model, train_loader, train_chunks, blocks, loss_margin, distance_type, dataset_share=0.25, augmentation_scale=augmentation_scale)

        print("Training loss:", train_loss, "| Validation loss:", val_loss)
        print("Training accuracy:", train_acc, "| Validation accuracy:", val_acc)
        
        if len(metrics["val_loss"]) > 0:
            if val_loss < min(metrics["val_loss"]): # save model with lowest error on validation set
                print("Saved best model")
                save_model(model, name + "_best_val")
        else:
            print("Saved best model")
            save_model(model, name + "_best_val")

        save_model(model, name) # save model after every epoch
        
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_acc"].append(val_acc)
       
        with open(os.path.join(losses_path, name + ".json"), "w") as lf:
            json.dump(metrics, lf)
        
    tracker.stop()
    del tracker

    return metrics


def get_metrics(model, data_loader, chunks, blocks, loss_margin, distance_type="euclidean", dataset_share=1, augmentation_scale=1):
    
    if not dataset_share == 1:
        rand_idx = torch.randperm(len(data_loader.dataset))
        rand_dataset = data_loader.dataset[rand_idx]
        rand_dataset = rand_dataset[:int(dataset_share*len(rand_dataset))]
        data_loader = torch.utils.data.DataLoader(rand_dataset, batch_size=data_loader.batch_size, shuffle=True)

    out = prd.get_triplet_loader_output_ssl(model, data_loader, chunks, blocks, augmentation_scale=augmentation_scale)
    loss = prd.calc_triplet_loss(out, loss_margin, distance_type)
    acc = prd.get_triplet_accuracy(out, loss_margin, distance_type)
    
    return loss.item(), acc


def get_losses(metrics_dict, model, blocks, criterion, val_loader=None, val_chunks=None, validation=True, train_loader=None, train_chunks=None, training=False, augmentation_scale=1):
    metrics = deepcopy(metrics_dict)

    if validation:
        val_loss = calc_loss(model, val_loader, val_chunks, blocks, criterion, augmentation_scale)
        metrics["val_loss"].append(val_loss)
    if training:
        train_loss = calc_loss(model, train_loader, train_chunks, blocks, criterion, augmentation_scale)
        metrics["train_loss"].append(train_loss)
    
    return metrics


def get_accuracies(metrics_dict, model, blocks, loss_margin, val_chunks=None, validation=True, train_chunks=None, training=False):
    metrics = deepcopy(metrics_dict)

    if validation:
        val_acc = prd.calc_triplet_accuracy_ssl(model, val_chunks, blocks, loss_margin)
        metrics["val_acc"].append(val_acc)
    if training:
        train_acc = prd.calc_triplet_accuracy_ssl(model, train_chunks, blocks, loss_margin)
        metrics["train_acc"].append(train_acc)
    
    return metrics


def calc_loss(model, data_loader, chunks, blocks, criterion, augmentation_scale=1):
    model.eval()
    model.to(device)
    losses = []

    with torch.no_grad():
        for batch_trp in data_loader:

            batch = pru.generate_triplet_data(batch_trp, chunks, blocks, scale=augmentation_scale)
            batch = batch.to(device)
            net_out = model(batch[:, 0, :, :], batch[:, 1, :, :], batch[:, 2, :, :])
            batch_loss = criterion(net_out[0], net_out[1], net_out[2])

            losses.append(batch_loss.item())

    return sum(losses) / len(losses)


def get_model_name(model_conf, val_sessions):
    model_name = f"conf_id_{model_conf.name}_model_{val_sessions[model_conf.models_trained]}_ssl"
    
    return model_name