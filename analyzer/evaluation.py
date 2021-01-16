"""
File containing methods for accuracy evaluations and grid searches of supervised and unsupervised methods.
"""
import pickle

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, silhouette_score, \
    balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split

from clustering.cluster import cluster_hierarchical, cluster_kmeans, cluster_mlp_autoencoder, cluster_hierarchical_n
from preprocessing.data_preprocessing import read_features, read_timeseries
from preprocessing.utils import to_distances, impute

np.random.seed(42)


# function that oversamples the folds created by the stratified Kfold
def oversample_folds(split, mort):
    new_split = []
    for train_idx, test_idx in split:
        ros = RandomOverSampler(random_state=0)
        oversampled_train, _ = ros.fit_resample(train_idx.reshape(-1, 1), mort[train_idx])
        np.random.shuffle(oversampled_train)
        new_split.append((oversampled_train.reshape(1, -1)[0], test_idx))
    return new_split


# scores used in evaluation of the clusters
def calculate_scores_clusters(labels, y_true, data, metric):
    print(labels)
    print(y_true)
    print(labels.shape, y_true.shape)
    silhouette = silhouette_score(data, labels, metric=metric)
    ari = adjusted_rand_score(y_true, labels)
    homogeneity = homogeneity_score(y_true, labels)
    completeness = completeness_score(y_true, labels)
    accuracy = balanced_accuracy_score(y_true, labels)

    return [silhouette, ari, homogeneity, completeness, accuracy]


# complete evaluation of the clustering methodsm with regard to mortality
def cluster_evaluation_mort(dtw_m, features, mort):
    n_clusters = [2, 3, 4, 5, 6, 7]
    print('Shapes')
    print(dtw_m.shape)
    for clusters in n_clusters:
        print("Cluster #", clusters)
        labels_hierarchical = cluster_hierarchical(features, clusters)
        labels_kmeans = cluster_kmeans(features, clusters)
        labels_mlp_ae, _ = cluster_mlp_autoencoder(features, clusters, mort)
        labels_dtw_hier = cluster_hierarchical_n(dtw_m, clusters)
        results = print_results(features, mort, dtw_m, labels_hierarchical, labels_kmeans, labels_mlp_ae,
                                labels_dtw_hier)
    print(results)
    return results


def print_results(features, mort, dtw_m, labels_hierarchical, labels_kmeans, labels_mlp_ae, labels_dtw_hier):
    results = []
    result_hier = calculate_scores_clusters(labels_hierarchical, mort, features, 'euclidean')
    print("Hierarchical:")
    print_scores(result_hier)

    result_kmeans = calculate_scores_clusters(labels_kmeans, mort, features, 'euclidean')
    print("Kmeans:")
    print_scores(result_kmeans)

    result_mlp_ae = calculate_scores_clusters(labels_mlp_ae, mort, features, 'euclidean')
    print("MLP_AE:")
    print_scores(result_mlp_ae)

    result_dtw_hier = calculate_scores_clusters(labels_dtw_hier, mort, dtw_m, 'precomputed')
    print("DTW_hierarchical:")
    print_scores(result_dtw_hier)
    unique_elements, counts_elements = np.unique(labels_dtw_hier, return_counts=True)
    print("Cluster sizes:")
    print(np.asarray((unique_elements, counts_elements)))

    results.append([result_hier, result_kmeans, result_mlp_ae, result_dtw_hier])
    return results


def print_scores(result_hier):
    print("Accuracy: %0.2f" % (result_hier[4]))
    print("Silhouette score: %0.2f" % (result_hier[0]))
    print("ARI: %0.2f" % (result_hier[1]))
    print("homogeneity: %0.2f, completeness: %0.2f" % (result_hier[2], result_hier[3]))


def grid_search(model, param_dict, split, x_train, y_train):
    grid = GridSearchCV(estimator=model, param_grid=param_dict, cv=split, n_jobs=1,
                        scoring=('f1', 'balanced_accuracy', 'roc_auc'), refit='balanced_accuracy')
    grid_result = grid.fit(x_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_balanced_accuracy']
    stds = grid_result.cv_results_['std_test_balanced_accuracy']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid_result


# grid search for MLP autoencoder
def grid_search_mlp_ae(x_train, y_train):
    neurons_h = [64, 32, 128]
    neurons_e = [8, 16, 32]
    batch_sizes = [10, 64, 256]
    epochs = [100, 200, 500]

    results = np.empty((0, 9))
    for neuron_h in neurons_h:
        for neuron_e in neurons_e:
            for batch_size in batch_sizes:
                for epoch in epochs:
                    sub_results = np.empty((0, 9))
                    for i in range(3):
                        y_pred = cluster_mlp_autoencoder(x_train, 2, y_train, neuron_h, neuron_e, epoch,
                                                         batch_size)
                        result_mlp_ae = calculate_scores_clusters(y_pred, y_train, x_train, 'euclidean')
                        result_mlp_ae = np.reshape(result_mlp_ae, (1, -1))
                        result_mlp_ae = np.append(result_mlp_ae, [neuron_h, neuron_e, batch_size, epoch])
                        sub_results = np.append(sub_results, [result_mlp_ae], axis=0)
                    results = np.append(results, [np.mean(sub_results, axis=0)], axis=0)

                    # best = results[[np.where(results==np.amax(results[:,4], axis=0))[0]],:]
    results_sorted = results[results[:, 4].argsort()]

    df_results = pd.DataFrame(data=results_sorted,
                              columns=["silhouette", "ari", "homogeneity", "completeness", "accuracy", "neuron_h",
                                       "neuron_e", "batch_size", "epoch"])
    print("Full results: ")
    print(df_results)

    return results_sorted


def split_test_train(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1, stratify=y)
    return x_train, y_train, x_test, y_test


# determine optimal cluster size using silhouette score (or another metric)
def determine_cluster_size(model, data, metric, distance=None):
    n_clusters = np.arange(2, 5)

    n_max = 0
    s_max = 0
    for n in n_clusters:
        y_pred = model(data, n)
        if distance is None:
            sil_score = silhouette_score(data, y_pred, metric=metric)
        else:
            sil_score = silhouette_score(distance, y_pred, metric=metric)

        print("For n_clusters =", n,
              ", the average silhouette_score is :", sil_score)

        if sil_score > s_max:
            n_max = n
            s_max = sil_score

    print("Optimal n = %s with silhouette score %0.2f" % (n_max, s_max))
    return n_max


if __name__ == "__main__":
    pred, ts, dtw_m, _ = read_timeseries()
    features, mort, _, feature_names, descriptives = read_features()
    gridsearch_on = False

    features = impute(features)

    pred = np.where(np.isnan(pred), -1, pred)
    print(features.shape, mort.shape, pred.shape)
    feat_train, y_train, feat_test, y_test = split_test_train(features, mort)
    ts_train, y_ts_train, feat_test, y_ts_test = split_test_train(pred, mort)

    if gridsearch_on:
        mlp_gp = cluster_mlp_autoencoder(ts_train, 6, mort)
        result_mlp_ae = calculate_scores_clusters(mlp_gp, mort, features, 'euclidean')
        print("MLP_AE:")
        print("Accuracy: %0.2f" % (result_mlp_ae[4]))
        print("Silhouette score: %0.2f" % (result_mlp_ae[0]))
        print("ARI: %0.2f" % (result_mlp_ae[1]))
        print("homogeneity: %0.2f, completeness: %0.2f" % (result_mlp_ae[2], result_mlp_ae[3]))
    #
    # # CALCULATE (AND SAVE) THE DTW DISTANCE MATRIX:
    # dtw_m = to_distances(ts)
    #
    # with open('../data/cdist_cluster_full.pkl', 'wb') as cdist_file:
    #     pickle.dump(dtw_m, cdist_file)

    # GRID SEARCH MLP AE:
    # grid_search_res_mlp_ae = grid_search_mlp_ae(feat_train, y_train)

    # with open('result_grid_MLP_AE.pkl', 'wb') as res_MLP_AE:
    #     pickle.dump(grid_search_res_mlp_ae, res_MLP_AE)

    # UNSUPERVISED ALGOS
    results_unsupervised = cluster_evaluation_mort(dtw_m, features, mort)
    with open('results_unsupervised.pkl', 'wb') as res_unsup:
        pickle.dump(results_unsupervised, res_unsup)
