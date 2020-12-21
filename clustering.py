import argparse
import os
import pandas as pd
import numpy as np

from analyzer.evaluation import determine_cluster_size
from analyzer.external_validation import clf_stability_analysis
from analyzer.internal_validation import validate_clustering
from clustering.cluster import cluster_hierarchical_n, cluster_kmeans, cluster_mlp_autoencoder
from preprocessing.data_preprocessing import read_timeseries, read_features
from preprocessing.utils import impute

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clustering options")
    parser.add_argument('--internal_validation', default=True, help='True or False')
    parser.add_argument('--external_validation', default=True, help='True or False')
    parser.add_argument('--exp_name', help='exp name')
    args = parser.parse_args()

    internal_val = args.internal_validation
    external_val = args.external_validation

    # extract data for training
    _, _, dtw_m, _ = read_timeseries()
    features, _, _, feature_names, descriptives = read_features()
    features = impute(features)

    # run models
    models = [
        [cluster_hierarchical_n, dtw_m, 'precomputed', 'HC'],
        [cluster_kmeans, features, 'euclidean', 'Kmeans'],
        [cluster_mlp_autoencoder, features, 'euclidean', 'MLP'],
    ]
    dtw_on = False
    for model in models:
        # if using optimal cluster sizes:
        if dtw_on is True:
            distance = dtw_m if model[2] == 'precomputed' else None
            n_clusters = determine_cluster_size(model[0], model[1], model[2], distance)

        n_clusters = [2, 3, 4, 5, 6]

        for n_cluster in n_clusters:
            y_pred = model[0](model[1], n_cluster)

            descriptives["cluster"] = y_pred
            descriptives.to_csv(
                '../stats/descriptives_cluster_val_cat' + str(model[0].__name__) + '_' + str(n_cluster) + '.csv',
                index=False)
            print(descriptives['cluster'].value_counts())
            print(model[3])

            if internal_val is True:
                n_trials = 1000
                cluster_results = np.empty([features.shape[0], n_trials])
                cluster_results_proba = np.empty([features.shape[0], n_clusters, n_trials])
                print("size results: " + str(np.size(cluster_results)))
                validate_clustering(features, models)

            if external_val is True:
                clf_stability_analysis(n_cluster)
