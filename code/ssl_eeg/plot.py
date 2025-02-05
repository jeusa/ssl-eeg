"""Plotting loss, accuracy, confusion matrix, dimension reduction."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from . import model, metrics

root_dir = os.getcwd().split("code")[0]
plots_path = os.path.join(root_dir, "plots")
conf_matrix_path = os.path.join(plots_path, "confusion_matrix")
dim_red_path = os.path.join(plots_path, "dimension_reduction")

n_main_color = "#a52744"
n_sec_color = "#d6788e"

u_main_color = "#1a3796"
u_sec_color = "#8397da"

n_color_cycle = sns.color_palette([
    "#f3b1c0",
    "#18a890",
    "#800e28",
    "#232d8d",])

plt.style.use(os.path.join(root_dir, "code", "ssl_eeg", "style.mplstyle"))

def plot_metrics(conf_ids, metric_key="val_loss", head=False, show_legend=True, y_title=None):
    if head:
        models_doc = pd.read_csv(model.models_doc_head_path, index_col=0)
    else:
        models_doc = pd.read_csv(model.models_doc_path, index_col=0)

    if y_title == None:
        if metric_key == "val_loss":
            y_title = "Validierungs-Loss"
        elif metric_key == "train_loss":
            y_title = "Trainings-Loss"
        elif metric_key == "train_acc":
            y_title = "Trainings-Triplet-Accuracy"
        elif metric_key == "val_acc":
            y_title = "Validierungs-Triplet-Accuracy"
        elif metric_key == "train_batch_loss":
            return
    
    fig, ax = plt.subplots()
    ax.set_xlabel("Epochen")
    ax.set_ylabel(y_title)

    if type(conf_ids) != list:
        conf_ids = [conf_ids]
        
    if len(conf_ids) > 1:
        show_legend = False

    for conf_id in conf_ids:
        for i, m in models_doc.loc[models_doc["conf_id"]==conf_id].iterrows():
            val_ses = m["val_idx"]
            mtr = metrics.load_loss_by_id(conf_id, val_ses, head)

            label_text = f"Trainings-Validierungs-Split {val_ses+1}"
            if "ssl" in m["model_name"]:
                label_text = f"Validierungsaufnahme Nr. {val_ses}"
            ax.plot(list(range(1,len(mtr[metric_key])+1)), mtr[metric_key], label=label_text)

    if show_legend:
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_xticks(list(range(1,len(mtr[metric_key])+1)))

    return fig


def plot_margin_accuracies_ssl(conf_id, data="unlabeled", val_ses=None):
    models_doc = pd.read_csv(model.models_doc_path, index_col=0)

    fig, ax = plt.subplots()
    ax.set_xlabel("loss margin")
    ax.set_ylabel("triplet validation accuracy")

    if not val_ses == None:
        m = models_doc.loc[(models_doc["conf_id"]==conf_id) & (models_doc["val_idx"]==val_ses)].iloc[0]
        marg_acc_u = metrics.load_margin_accuracies(m["model_name"], set="validation")
        marg_acc2_l= metrics.load_margin_accuracies(m["model_name"], set="test")
        marg_acc3_l= metrics.load_margin_accuracies(m["model_name"], set="disjoint_test")


        ax.plot(marg_acc_u["margins"], marg_acc_u["accuracies"], label=f"unlabeled data", color=u_main_color)
        ax.plot(marg_acc2_l["margins"], marg_acc2_l["accuracies"], label=f"n-back data", color=n_main_color)
        ax.plot(marg_acc3_l["margins"], marg_acc3_l["accuracies"], label=f"n-back data disjoint", color=n_sec_color)
    else:
        for i, m in models_doc.loc[models_doc["conf_id"]==conf_id].iterrows():
            marg_acc = metrics.load_margin_accuracies(m["model_name"], set="validation")
            val_ses = m["val_idx"]
            ax.plot(marg_acc["margins"], marg_acc["accuracies"], label=f"validation session no. {val_ses}")

    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()

    return fig


def plot_conf_matrix(prediction=None, labels_true=None, labels_pred=None, cm=None, german=True):
    mpl.rcParams["axes.grid"] = False

    if not type(cm) is np.ndarray:
        if type(prediction) is pd.DataFrame:
            labels = prediction[["n", "pred_n"]].to_numpy()
        else:
            labels = np.stack([labels_true, labels_pred], axis=1)
        cm = confusion_matrix(labels[:,0], labels[:, 1], normalize="true")
        
    disp = ConfusionMatrixDisplay(cm*100)
    disp.plot(values_format=".2f")
    if german:
        disp.ax_.set_xlabel("Vorhergesagte Klasse")
        disp.ax_.set_ylabel("Wahre Klasse")
    mpl.rcParams["axes.grid"] = True

    return cm, disp


def plot_dim_red(red_data, hue_data, palette=sns.color_palette(), legend_title="", hue_order=None, size=20, save=False, file_name=None, german=True):

    ax = sns.scatterplot(x=red_data[:,0], y=red_data[:,1], hue=hue_data, s=size, palette=palette, hue_order=hue_order)
    plt.legend(title=legend_title)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.04, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if save:
        plt.savefig(os.path.join(dim_red_path, file_name + ".png"), dpi=200, bbox_inches="tight")
        plt.show()
    else:
        plt.show()

    return ax


def save_plot_and_legend(axis, file_path):
    fig = axis.figure
    box_legend = axis.get_legend().get_window_extent().transformed(fig.dpi_scale_trans.inverted()).get_points()
    box_fig = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).get_points()

    fig.savefig(os.path.join(plots_path, file_path + "_legend.png"), dpi=200, bbox_inches=mpl.transforms.Bbox([[box_legend[0][0], 0], [box_legend[1][0], box_fig[1][1]]]))
    axis.get_legend().remove()
    fig.savefig(os.path.join(plots_path, file_path + ".png"), dpi=200, bbox_inches="tight")