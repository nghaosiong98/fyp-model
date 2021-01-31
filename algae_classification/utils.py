import matplotlib.pyplot as plt
import os
import seaborn as sn
import pandas as pd


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def save_loss_graph(hist, path):
    plt.figure()
    plt.ylabel("Loss (training and validation)")
    plt.xlabel("Training Steps")
    plt.ylim([0, 2])
    plt.plot(hist["loss"], label='train')
    plt.plot(hist["val_loss"], label='val')
    plt.legend()
    plt.savefig(os.path.join(path, 'loss.png'))


def save_accuracy_graph(hist, path):
    plt.figure()
    plt.ylabel("Accuracy (training and validation)")
    plt.xlabel("Training Steps")
    plt.ylim([0, 1])
    plt.plot(hist["categorical_accuracy"], label='train')
    plt.plot(hist["val_categorical_accuracy"], label='val')
    plt.legend()
    plt.savefig(os.path.join(path, 'accuracy.png'))


def save_confussion_matrix(cm, path, index, columns):
    df_cm = pd.DataFrame(cm, index=index, columns=columns)
    plt.figure(figsize=(10, 7))
    cm_plot = sn.heatmap(df_cm, annot=True)
    figure = cm_plot.get_figure()
    figure.savefig(os.path.join(path, 'cm.png'))
