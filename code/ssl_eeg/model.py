"""Definition of siamese EEGNet and the classifier. Loading and saving a model."""

import torch
import pandas as pd
import os
from torch import nn

from .preprocessing import s_length

root_dir = os.getcwd().split("code")[0]
models_path = os.path.join(root_dir, "models")
models_conf_path = os.path.join(models_path, "models_conf.csv")
models_conf_head_path = os.path.join(models_path, "models_conf_head.csv")
models_doc_path = os.path.join(models_path, "models_doc.csv")
models_doc_head_path = os.path.join(models_path, "models_doc_head.csv")

class SiameseEegNet(nn.Module):

    def __init__(self, out_dim, dropout_p):
        f1 = 8
        d = 2
        f2 = f1*d
        c = 8 # number eeg channels

        super(SiameseEegNet, self).__init__()

        # channelwise convolution
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=f1, kernel_size=(1, 125), stride=1, padding="same", bias=False),
            nn.BatchNorm2d(num_features=f1)
        )

        # depthwise convolution
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=f1, out_channels=f1*d, kernel_size=(c, 1), groups=f1, stride=1, padding="valid", bias=False),
            nn.BatchNorm2d(num_features=f1*d),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,4)),
            nn.Dropout(p=dropout_p)
        )

        # separable depthwise convolution
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=f1*d, out_channels=f1*d, kernel_size=(1, 31), groups=f1*d, stride=1, padding="same", bias=False),
            nn.Conv2d(in_channels=f1*d, out_channels=f2, kernel_size=1, stride=1, padding="valid", bias=False),
            nn.BatchNorm2d(num_features=f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,8)),
            nn.Dropout(p=dropout_p)
        )

        self.block4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=f2*(s_length//32), out_features=out_dim)
        )


    def forward_once(self, x):
        output = x

        if output.dim() == 2:
            output = torch.reshape(output, (1, output.shape[0], output.shape[1]))

        output = torch.reshape(output, (output.shape[0], 1, output.shape[1], output.shape[2]))
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)

        return output

    def forward(self, anchor, pos, neg):
        anchor_emb = self.forward_once(anchor)
        pos_emb = self.forward_once(pos)
        neg_emb = self.forward_once(neg)

        return anchor_emb, pos_emb, neg_emb


class HeadNet(nn.Module):

    def __init__(self, in_dim=64):
        super(HeadNet, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=2*in_dim),
            nn.ReLU(),
            nn.Linear(in_features=2*in_dim, out_features=in_dim),
            nn.ReLU(),
            nn.Linear(in_features=in_dim, out_features=4)
        )
    
    def forward(self, x):
        output = self.head(x)

        return output


def save_model(model, model_name):
    torch.save(model.state_dict(), os.path.join(models_path, model_name))


def load_model(model_name, out_dim, dropout_p):
    model = SiameseEegNet(out_dim, dropout_p)
    
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(os.path.join(models_path, model_name)))
    else:
        model.load_state_dict(torch.load(os.path.join(models_path, model_name), map_location=torch.device('cpu')))

    model.eval()

    return model


def load_head_model(model_name, in_dim):
    model = HeadNet(in_dim=in_dim)
    
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(os.path.join(models_path, model_name)))
    else:
        model.load_state_dict(torch.load(os.path.join(models_path, model_name), map_location=torch.device('cpu')))

    model.eval()

    return model


def get_same_config(conf_id):
    models_conf = pd.read_csv(models_conf_path, index_col=0)
    end_idx = list(models_conf.columns).index("models_trained")
    models_conf_r = models_conf[models_conf.columns[:end_idx].drop("epochs")]
    cur_conf_r = models_conf_r.loc[conf_id]

    if conf_id in models_conf.index:
        for i, row in models_conf_r.loc[:conf_id-1].sort_index(ascending=False).iterrows():
            if cur_conf_r.compare(row).empty:
                base_id = i
                return base_id
    
    return 0
