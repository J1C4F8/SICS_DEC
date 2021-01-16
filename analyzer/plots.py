"""
This file can be used to generate plots of the gaussian Process Regression
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import random

from seaborn.matrix import dendrogram
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, WhiteKernel, ConstantKernel as C
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from preprocessing.data_preprocessing import read_sics, read_extra
from preprocessing.variables import variables

font = {'family': 'serif',
        'weight': 'normal',
        'size': 16,
        }

sns.set_style("dark")
sns.set_palette(sns.color_palette("mako_d", 40))


# sns.set_context("talk")


def convert_to_reltime(t_lab, los):
    start = t_lab.min()
    t_lab = t_lab - start
    t_lab = t_lab / np.timedelta64(1, 'h')
    max_hours = los[los['studyID'] == t_lab.name]['LOS_hours'].iloc[0]
    t_lab = t_lab.apply(lambda x: x if x <= max_hours else -1)

    return t_lab


def plot_raw(studyID):
    sics_raw = extract_sics_raw(studyID)
    sics_raw = sics_raw[["value_name", "Reltime"]]
    sics_raw = sics_raw.sort_values(by=["value_name"])

    sns.catplot(x="Reltime", y="value_name", jitter=False, data=sics_raw, marker='X', height=8, aspect=2.5)
    plt.xlabel("$time (hours)$")
    plt.ylabel("$variable name$")
    plt.title("Measurements in the SICS dataset")
    plt.tight_layout()

    plt.savefig("raw_data_plot.eps", format='eps', dpi=1000)
    plt.close()


def plot_raw_vars(studyID):
    sics_raw = extract_sics_raw(studyID)
    sics_raw = sics_raw.sort_values(by=["value_name"])

    variables = sics_raw["value_name"].unique()
    fig, ax = plt.subplots(figsize=(16, 9), ncols=8, nrows=5)

    plt.subplots_adjust(
        wspace=0.4,
        hspace=0.8
    )

    count = 0
    for var in variables:
        axis = ax[count % 5][count % 8]
        los = sics_raw["Reltime"].max()
        sics_sub = sics_raw[sics_raw["value_name"] == var]
        y = sics_sub["value"]
        x = sics_sub["Reltime"]
        sns.scatterplot(x, y, ax=axis)
        axis.set_title(var)
        axis.set_xlim(left=0, right=los)
        axis.set_xlabel('$value$')
        axis.set_ylabel('$time (hours)$')
        plt.suptitle("Values in the SICS dataset")
        count += 1

    plt.savefig("raw_vars_plot.eps", format='eps', dpi=1000)
    plt.close()


def plot_gaussian(y_pred, y, sigma, X, x2):
    fig, ax = plt.subplots(figsize=(16, 9), ncols=2, nrows=1)
    print(X, y)
    sns.scatterplot(X, y, ax=ax[0])
    sns.scatterplot(X, y, ax=ax[1])
    sns.lineplot(np.arange(len(y_pred)), y_pred, ax=ax[1])
    ax[1].fill(np.concatenate([x2, x2[::-1]]),
               np.concatenate([y_pred - 1.9600 * sigma,
                               (y_pred + 1.9600 * sigma)[::-1]]),
               alpha=.5, ec='None', label='95% confidence interval')
    ax[0].set_xlabel('$time (hours)$')
    ax[0].set_ylabel('$value$')
    ax[1].set_xlabel('$time (hours)$')
    ax[1].set_ylabel('$value$')

    plt.savefig("gauss_plot.eps", format='eps', dpi=1000)
    plt.close()


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')


def plot_clusters(y, prediction, var):
    print(prediction)
    plt.figure()
    sz = y.shape[1]
    plt.suptitle("Hierarchical clustering " + variables[var]['Name'] + ", " + str(y.shape[0]) + " patients")
    nr_clusters = np.size(np.unique(prediction))
    print(nr_clusters)
    for yi in range(nr_clusters):
        plt.subplot(1, nr_clusters, yi + 1)
        for xx in y[prediction == yi]:
            plt.plot(xx[:, 0], "b-", alpha=.2)
        plt.xlim(0, sz)
        plt.ylim(np.nanmin(y), np.nanmax(y))
        plt.xlabel('$time (hours)$')


def extract_sics_raw(studyID):
    sics_raw = read_sics()
    base = read_extra()
    sics_raw = sics_raw[sics_raw["studyID"] == studyID]
    sics_raw["Reltime"] = sics_raw.groupby(["studyID"])["t_lab"].apply(lambda x: convert_to_reltime(x, base))
    sics_raw = sics_raw[sics_raw["Reltime"] != -1.0]
    sics_raw = sics_raw[["value_name", "value", "Reltime"]]
    return sics_raw


def plot_dendrogram(Z):
    plt.figure(figsize=(25, 10))
    dendrogram(Z)


def plot_mean(X, y, x2, y_pred):
    plt.figure()
    plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
    plt.plot(x2, y_pred, 'b-', label=u'Prediction')
    plt.xlabel('$time (hours)$')
    plt.ylabel('$value$')
    plt.legend(loc='upper left')


def plot_data(x, y):
    plt.plot(x, y, 'bo', label="Ureum")
    plt.xlabel('$time$ $(hours)$')
    plt.ylabel('$value$ $(mmol/L)$')
    plt.legend(loc='upper left')


def plot_confusion_matrix(y_true, y_pred, normalize=False,
                          title=None, cmap='Blues'):
    print(type(y_pred))
    y_true = pd.Categorical(pd.Series(y_true)).rename_categories({0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}).astype('int64')
    y_pred = pd.Categorical(pd.Series(y_pred)).rename_categories({0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}).astype('int64')
    classes = unique_labels(y_true, y_pred)
    print(y_true)
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    print(unique_labels(y_true, y_pred), 'unique labels')
    # classes = classes[unique_labels(y_true, y_pred)]
    print(classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
