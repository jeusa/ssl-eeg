"""Train a supervised encoder."""

import torch
import numpy as np
import json
import os
from tqdm.auto import trange
from torch import nn
from codecarbon import EmissionsTracker

from . import preprocessing_nback as prn, prediction as prd
from .model import models_path, save_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device is: {device}")

losses_path = os.path.join(models_path, "losses")


def train(model, epochs, chunks_data, train_chunks_df, val_chunks_df, batch_size, val_batch_size, name, learning_rate=0.001, loss_margin=1, distance_type="euclidean"):
    tracker = EmissionsTracker(project_name=name, output_dir=models_path, log_level="warning")
    tracker.start()
    
    # distance function
    dis_f = None
    if distance_type == "euclidean":
        dis_f = nn.PairwiseDistance(p=2)
    elif distance_type == "cosine":
        dis_f = lambda x, y: 1.0 - torch.nn.functional.cosine_similarity(x, y, dim=0)

    model.to(device)
    criterion = nn.TripletMarginWithDistanceLoss(distance_function=dis_f, margin=loss_margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = {
        "train_batch_loss": [],
        "train_loss": [],
        "val_loss": []
    }

    for epoch in trange(epochs, desc="Training"):
        # set new random negative samples
        train_triplets = prn.make_triplets(train_chunks_df)
        val_triplets = prn.make_triplets(val_chunks_df)

        train_loader = torch.utils.data.DataLoader(train_triplets, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_triplets, batch_size=val_batch_size, shuffle=True)
        train_it = iter(train_loader)
        max_steps = int(np.ceil(train_triplets.shape[0] / batch_size))
        
        # training
        model.train()
        for b in trange(max_steps, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            
            batch_trp = next(train_it)
            batch = prn.generate_data_from_triplets(batch_trp, chunks_data)
            batch = batch.to(device)
            
            net_out = model(batch[:, 0, :, :], batch[:, 1, :, :], batch[:, 2, :, :])

            batch_loss = criterion(net_out[0], net_out[1], net_out[2])
            batch_loss.backward()
            optimizer.step()

            losses["train_batch_loss"].append(batch_loss.item())
        
        # calc loss after every epoch
        val_loss = calc_loss_loader(model, val_loader, chunks_data, criterion)
        train_loss = calc_loss_loader(model, train_loader, chunks_data, criterion, dataset_share=0.2)
            
        print("Training loss:", train_loss, "| Validation loss:", val_loss)
        
        if len(losses["val_loss"]) > 0:
            if val_loss < min(losses["val_loss"]): # save model with lowest error on validation set
                print("Saved best model")
                save_model(model, name + "_best_val")
        else:
            print("Saved best model")
            save_model(model, name + "_best_val")
        
        save_model(model, name) # save model after every epoch

        losses["val_loss"].append(val_loss)
        losses["train_loss"].append(train_loss)
        
        with open(os.path.join(losses_path, name + ".json"), "w") as lf:
            json.dump(losses, lf)
        
    tracker.stop()
    del tracker

    return losses


def calc_loss_loader(model, data_loader, chunks_data, criterion, dataset_share=1):
    if not dataset_share == 1:
        rand_idx = torch.randperm(len(data_loader.dataset))
        rand_dataset = data_loader.dataset[rand_idx]
        rand_dataset = rand_dataset[:int(dataset_share*len(rand_dataset))]
        data_loader = torch.utils.data.DataLoader(rand_dataset, batch_size=data_loader.batch_size, shuffle=True)

    out = prd.get_triplet_loader_output_sl(model, data_loader, chunks_data)
    loss = criterion(out[:,0], out[:,1], out[:,2]).item()

    return loss


def get_model_name(model_conf):
    model_name = f"conf_id_{model_conf.name}_model_{int(model_conf.models_trained+1)}"
    
    return model_name


