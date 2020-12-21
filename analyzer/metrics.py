"""
File containing metrics for DEC clusterin, as implemented by https://github.com/Tony607/Keras_Deep_Clustering
"""

import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import keras.backend as K

from sklearn.utils.linear_assignment_ import linear_assignment


# compute sensitivity, can be used in Keras evaluation
def sensitivity(y_true, y_pred):
    """

    Parameters
    ----------
    y_true :
    y_pred :

    Returns
    -------

    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


# compute specificity, can be used in Keras evaluation
def specificity(y_true, y_pred):
    """

    Parameters
    ----------
    y_true :
    y_pred :

    Returns
    -------

    """
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def calculate_metrics(loss, y, y_pred):
    """

    Parameters
    ----------
    loss :
    y :
    y_pred :

    Returns
    -------

    """
    nmi = normalized_mutual_info_score
    ari = adjusted_rand_score
    acc = np.round(accuracy(y_true=y, y_pred=y_pred), 5)
    nmi = np.round(nmi(y, y_pred), 5)
    ari = np.round(ari(y, y_pred), 5)
    loss = np.round(loss, 5)
    return acc, ari, loss, nmi


def accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy
    Parameters
    ----------
    y_true : true labels, numpy.array with shape `(n_samples,)`
    y_pred : predicted labels, numpy.array with shape `(n_samples,)`

    Returns
    -------
    accuracy: array with shape [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
