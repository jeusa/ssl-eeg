"""Train a classifier, calculate loss and classification accuracy for it."""

import json
import torch
import numpy as np
import os

from torch import nn
from sklearn.metrics import accuracy_score
from tqdm.auto import trange
from codecarbon import EmissionsTracker

from .model import save_model, models_path
from .train_sl import losses_path, device


def train(model, epochs, train_features, train_y, val_features, val_y, batch_size, val_batch_size, name, learning_rate=0.001, metrics=None):
    tracker = EmissionsTracker(project_name=name, output_dir=models_path, log_level="warning")
    tracker.start()

    train_data = torch.concat([train_features, torch.reshape(train_y, (train_y.shape[0], 1))], dim=1)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data = torch.concat([val_features, torch.reshape(val_y, (val_y.shape[0], 1))], dim=1)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=val_batch_size, shuffle=True)

    model.to(device)
    criterion = nn.CrossEntropyLoss() # includes softmax
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    max_steps = int(np.ceil(train_features.shape[0] / batch_size))

    if metrics == None:
        metrics = {
            "train_batch_loss": [],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

    for epoch in trange(epochs, desc="Training"):
        print(f"Epoch {epoch+1}/{epochs}")

        train_it = iter(train_loader)
        model.train()

        for b in trange(max_steps, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()

            batch = next(train_it)
            batch = batch.to(device)
            batch_feat = batch[:,:-1]
            batch_y = batch[:,-1]
            batch_y = batch_y.to(torch.long)

            net_out = model(batch_feat)

            batch_loss = criterion(net_out, batch_y)
            batch_loss.backward()
            optimizer.step()

            metrics["train_batch_loss"].append(batch_loss.item())
        
        # calc loss on validation set after every epoch
        val_loss = calc_loss(model, val_loader, criterion)
        train_loss = calc_loss(model, train_loader, criterion)
        val_acc = calc_accuracy(model, val_features, val_y)
        train_acc = calc_accuracy(model, train_features, train_y)

        print("Training loss:", train_loss, "| Validation loss:", val_loss)
        
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


def calc_loss(model, data_loader, criterion):
    model.eval()
    model.to(device)
    loss = []

    with torch.no_grad():
        for batch in data_loader:

            batch = batch.to(device)
            batch_feat = batch[:,:-1]
            batch_y = batch[:,-1]
            batch_y = batch_y.to(torch.long)

            net_out = model(batch_feat)
            batch_loss = criterion(net_out, batch_y)

            loss.append(batch_loss.item())

    return sum(loss) / len(loss)


def make_predictions(model, features, prob=False):
    model.eval()
    model.to(device)
    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        out = model.forward(features)
        pred_prob = softmax(out)
        pred_y_val = torch.argmax(pred_prob, dim=1)
    
    if prob:
        return pred_prob
    else:
        return pred_y_val
    
def calc_accuracy(model, features, true_y):
    pred = make_predictions(model, features)

    return accuracy_score(true_y.cpu(), pred.cpu())