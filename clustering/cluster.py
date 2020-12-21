"""
File containing methods for unsupervised clustering of time series
"""
import random

import numpy as np
import scipy.spatial.distance as ssd
from keras import regularizers, Sequential
from keras.layers import Dense, Dropout
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import classification_report
from tslearn.metrics import cdist_dtw
from tslearn.utils import to_time_series, to_time_series_dataset

from analyzer.metrics import calculate_metrics
from analyzer.plots import plot_confusion_matrix, plot_loss
from preprocessing.gen_input import get_gaussian, select_data
from preprocessing.variables import variables

random.seed(1)

from clustering.DECLayer import DECLayer


def to_timeseries_set(df, var):
    ids = list(df['studyID'].unique())
    selected_ids = random.sample(ids, 10)
    print(len(ids))
    ts = []
    empty = 0
    count = 0
    for Id in selected_ids:
        count += 1
        sub_df, stop = select_data(df, Id, var)
        if len(sub_df.index) > 1:
            x = sub_df.loc[sub_df['value_name'] == variables[var]['Name']]['Reltime']
            y = sub_df.loc[sub_df['value_name'] == variables[var]['Name']]['value']
            _, y_pred, _ = get_gaussian(x, y, stop, var)
            y_pred = to_time_series(y_pred)
            ts.append(y_pred)

    dat = to_time_series_dataset(ts)
    print(dat.shape)
    print("nr ids: ", count)
    return dat, empty


def to_distances(ts_dataset):
    m = cdist_dtw(ts_dataset)
    m = ssd.squareform(m)
    return m


def cluster_hierarchical_n(d_matrix, n_classes):
    clustering = AgglomerativeClustering(n_clusters=n_classes, affinity='precomputed', linkage='complete').fit(d_matrix)
    return clustering.labels_


def cluster_hierarchical(feature_array, n_classes):
    clustering = AgglomerativeClustering(n_clusters=n_classes, linkage='complete').fit(feature_array)
    return clustering.labels_


def cluster_kmeans(feature_array, n_classes):
    clustering = KMeans(n_clusters=n_classes).fit(feature_array)
    return clustering.labels_


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


def dec_cluster(encoder, feature_array, y, n_classes):
    clustering_layer = DECLayer(n_classes, name='clustering')(encoder.output)
    model = Model(inputs=encoder.input, outputs=clustering_layer)
    model.compile(optimizer=SGD(0.01, 0.9), loss='kld')

    kmeans = KMeans(n_clusters=n_classes, n_init=20)
    y_pred = kmeans.fit_predict(encoder.predict(feature_array))
    y_pred_last = np.copy(y_pred)

    model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

    loss = 0
    index = 0
    maxiter = 8000
    batch_size = 256
    update_interval = 140
    index_array = np.arange(feature_array.shape[0])

    tol = 0.01  # tolerance threshold to stop training

    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            q = model.predict(feature_array, verbose=0)
            p = target_distribution(q)  # update the auxiliary target distribution p

            # evaluate the clustering performance
            y_pred = q.argmax(1)
            if y is not None:
                acc, ari, loss, nmi = calculate_metrics(loss, y, y_pred)
                print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

            # check stop criterion - model convergence
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            if ite > 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print('Reached tolerance threshold. Stopping training.')
                break
        idx = index_array[index * batch_size: min((index + 1) * batch_size, feature_array.shape[0])]
        loss = model.train_on_batch(x=feature_array[idx], y=p[idx])
        index = index + 1 if (index + 1) * batch_size <= feature_array.shape[0] else 0

    q = model.predict(feature_array, verbose=0)
    p = target_distribution(q)  # update the auxiliary target distribution p

    # evaluate the clustering performance
    y_pred = q.argmax(1)
    if y is not None:
        acc, ari, loss, nmi = calculate_metrics(loss, y, y_pred)
        print('Acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari), ' ; loss=', loss)

    return y_pred, q


def cluster_mlp_autoencoder(feature_array, n_classes, y=None, neurons_h=64, neurons_e=8, epochs=500, batch_size=64):
    input_arr = Input(shape=(feature_array.shape[1],))
    encoded = Dense(neurons_h, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_arr)
    encoded = Dense(neurons_e, activation='relu')(encoded)

    decoded = Dense(neurons_h, activation='relu')(encoded)
    decoded = Dense(feature_array.shape[1], activation='sigmoid')(decoded)

    autoencoder = Model(input_arr, decoded)
    encoder = Model(input_arr, encoded)

    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    autoencoder.fit(feature_array, feature_array,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True)

    y_pred, y_proba = dec_cluster(encoder, feature_array, y, n_classes)

    return y_pred, y_proba
