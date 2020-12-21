"""
File containing methods for computing cluster stability (average entropy over association probabilities)
"""
import itertools

import numpy as np
from scipy.stats import entropy

from clustering.cluster import cluster_hierarchical, cluster_kmeans, cluster_mlp_autoencoder
from preprocessing.data_preprocessing import extract_features
from preprocessing.utils import impute


def calculate_entropy(features, cluster_results, n_trials):
    """

    Parameters
    ----------
    features :
    cluster_results :
    n_trials :

    Returns
    -------

    """
    # calculate entropy for each pair
    pairs = itertools.combinations(np.arange(features.shape[0]), 2)
    pairs = list(pairs)
    n_pairs = len(pairs)
    pair_associations = np.empty(n_pairs)
    entropy_values = np.empty(n_pairs)
    for pair in range(len(pairs)):
        count = 0
        for trial in range(n_trials):
            if cluster_results[pairs[pair][0], trial] == cluster_results[pairs[pair][1], trial]:
                count += 1

        pair_associations[pair] = count / n_trials

        entropy_values[pair] = entropy([pair_associations[pair], 1 - pair_associations[pair]], base=2)
    average_entropy = sum(entropy_values) / n_pairs
    with open('../stats/average_entropy_1000.txt', 'w') as f:
        f.write(str(average_entropy) + '\n')
    print(average_entropy)


def calculate_cluster_results(cluster_results, cluster_results_proba, n_trials):
    """

    Parameters
    ----------
    cluster_results :
    cluster_results_proba :
    n_trials :

    Returns
    -------

    """
    np.savetxt("../stats/results_new_" + str(n_trials) + "_clusters_MLP_autoencoder_" + str(n_clusters) + ".csv",
               cluster_results, delimiter=",")
    print(cluster_results_proba.shape)
    cluster_results_proba = np.mean(cluster_results_proba, axis=2).argmax(1)
    print(cluster_results_proba.shape)
    np.savetxt("../stats/results_probas_" + str(n_trials) + "_clusters_MLP_autoencoder_" + str(n_clusters) + ".csv",
               cluster_results_proba, delimiter=",")


def validate_clustering(features, models):
    """

    Parameters
    ----------
    features :
    models :

    Returns
    -------

    """
    n_trials = 1000
    cluster_results = np.empty([features.shape[0], n_trials])
    cluster_results_proba = np.empty([features.shape[0], n_clusters, n_trials])
    print("size results: " + str(np.size(cluster_results)))

    for model in models:
        for trial in range(n_trials):
            y_pred, y_proba = model[0](model[1], n_clusters)
            cluster_results[:, trial] = y_pred
            cluster_results_proba[:, :, trial] = y_proba

        calculate_cluster_results(cluster_results, cluster_results_proba, n_trials)
        calculate_entropy(features, cluster_results, n_trials)


if __name__ == "__main__":
    features, _, _, feature_names, descriptives = extract_features()

    features = impute(features)

    models = [
        [cluster_hierarchical, features, 'euclidean', 'HC'],
        [cluster_kmeans, features, 'euclidean', 'Kmeans'],
        [cluster_mlp_autoencoder, features, 'euclidean', 'MLP'],
    ]

    n_clusters = 6

    validate_clustering(features, models)
