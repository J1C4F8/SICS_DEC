# gradient boosting
import os

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute._iterative import IterativeImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import xgboost as xgb

import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np

from analyzer.plots import plot_confusion_matrix


def classify(*arrays, directory, n_cluster, max_score=None, trial_no=None):
    """
    XGBoost clusters classifier
    Parameters
    ----------
    arrays : numpy arrays
        Train and test arrays
    directory : str
        Directory name
    n_cluster : int
        Cluster size
    max_score : float
        Maximum accuracy
    trial_no : str
        Trial number

    Returns
    ----------
    model: XGBoost model
    y_pred: numpy array
    max_score: float
    """
    x_train, y_train = arrays[0], arrays[1]
    x_test, y_test = arrays[2], arrays[3]
    cv_params = {
        'min_child_weight': [2],
        'gamma': [0.3],
        'subsample': [0.8],
        'colsample_bytree': [0.9],
        'max_depth': [5],
        'max_delta_step': [1],
        'learning_rate': [0.05],
        'n_estimators': [200],
    }
    grid_search_clf = GridSearchCV(xgb.XGBClassifier(random_state=0),
                                   cv_params, cv=10, n_jobs=-1)
    grid_search_clf.fit(x_train, y_train)
    print(grid_search_clf.best_estimator_)
    print(grid_search_clf.best_params_)

    model = grid_search_clf.best_estimator_
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(x_train.shape, x_test.shape, 'after predict')

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    if max_score is None or max_score < accuracy:
        save_results(directory, report, x_test, y_test, trial_no, n_cluster)
        max_score = accuracy
        print('Saving new max score=%.2f, trial_no=%s ...' % (accuracy, trial_no))
    else:
        print('------------SKIP trial_no=%s------------' % trial_no)
    return model, y_pred, max_score


def save_results(directory, report, x_test, y_test, trial_no, n_cluster):
    # Saving scores
    score_df = pd.DataFrame(report).transpose()
    score_df.to_csv(directory + '/scores.csv')

    # Saving results
    results_df = x_test.copy()
    results_df['true_cluster'] = y_test
    results_df.to_csv(directory + '/results.csv')

    # save raw table
    df = pd.read_csv('stats/sics.csv', index_col=0)
    clusters = pd.read_csv('stats/results_100_clusters_MLP_autoencoder_' + str(n_cluster) + '.csv',
                           names=[str(i) for i in range(1, 101)])
    df['cluster'] = clusters[trial_no]
    df.to_csv(directory + '/full_table.csv')


def generate_plots(directory, features, model, n_cluster, x_test, x_train, y_pred, y_test):
    # Plotting and saving confusion matrix
    print(len(y_test), len(y_pred))

    plot_confusion_matrix(y_test, y_pred)
    path = directory + '/confusion_matrix.png'
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.show()

    # SHAP plots
    shap_plots(directory, features, model, n_cluster, x_test, x_train)


def shap_plots(directory, features, model, n_cluster, x_test, x_train):
    booster = model.get_booster()
    model_bytearray = booster.save_raw()[4:]
    booster.save_raw = model_bytearray
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(x_test)

    print(shap_values.shape)
    print(x_train.shape, x_test.shape)

    save_shap(features, shap_values)

    # extract lab variables names
    features = extract_labs_names(features)

    if not os.path.exists(directory):
        os.makedirs(directory)

    save_shap_plots(directory, features, n_cluster, shap_values, x_test)


def save_shap_plots(directory, features, n_cluster, shap_values, x_test):
    shap.summary_plot(shap_values, x_test, feature_names=features, max_display=10, show=False)
    plt.savefig(directory + '/summary_plot_total.png', bbox_inches='tight', facecolor='white')
    plt.show()
    if n_cluster > 2:
        for idx, shap_value in enumerate(shap_values):
            shap.summary_plot(shap_value, x_test, feature_names=features, max_display=10, show=False)
            path = directory + '/summary_plot_cluster' + str(idx + 1) + '.png'
            plt.savefig(path, bbox_inches='tight', facecolor='white')
            plt.show()


def extract_labs_names(features):
    features_df = pd.read_csv('labs_names.csv')
    features_dict = pd.Series(features_df.Real.values, index=features_df.Raw).to_dict()
    features = [features_dict[feature.split('_')[0]] + '_' + feature.split('_', 1)[1] for feature in features
                if feature.split('_')[0] in features_dict]
    features = [feature.replace('_get_', ' - ') for feature in features]
    return features


def save_shap(features, shap_values):
    for idx in range(len(shap_values)):
        shap_values_df = pd.DataFrame(data=shap_values[idx], columns=features)
        shap_values_df.to_csv('../cluster_stability_results/6/shap_values' + str(idx + 1) + '.csv')


def group_shap_values(features, groups, groups_names, shap_values):
    new_features = []
    new_shap_values = []
    if isinstance(shap_values, list):
        for shap_value in shap_values:
            shap_df = extract_groups_of_features(features, groups, groups_names, shap_value)
            new_shap_value = shap_df.values
            new_shap_values.append(new_shap_value)
            new_features = shap_df.columns
    else:
        shap_df = extract_groups_of_features(features, groups, groups_names, shap_values)
        new_shap_values = shap_df.values
        new_features = shap_df.columns
    return new_shap_values, new_features


def extract_groups_of_features(features, groups, groups_names, shap_value):
    shap_df = pd.DataFrame(data=shap_value, columns=features)
    for idx, group in enumerate(groups):
        shap_df[groups_names[idx]] = np.nan
        for feature in group:
            spike_cols = [col for col in shap_df.columns if feature in col]
            shap_df[groups_names[idx]] = shap_df[spike_cols].sum(axis=1)
            shap_df = shap_df.drop(columns=spike_cols)
    return shap_df


def stratify_check(y):

    stratify_on = True
    for i in y.value_counts():
        if i < 2:
            stratify_on = False
            break
    return stratify_on


def prepare_data(df):
    if 'studyID' in df.columns:
        df = df.drop(columns=['studyID'])
    entry = df.index
    columns = df.columns

    # Scaling and imputing
    imp = IterativeImputer(max_iter=10, random_state=0)
    mat = imp.fit_transform(df.values)
    scaled_features = pd.DataFrame(index=entry, columns=columns, data=mat)
    return columns, scaled_features


def clf_stability_analysis(n_cluster):
    clusters = pd.read_csv('cluster_stability_results/' + str(n_cluster) + '/cluster_augmented_data_completed.csv',
                           index_col=0)
    df = pd.read_csv('stats/sics.csv', index_col=0)

    # Specifying directory
    directory_name = 'results'
    directory = directory_name + '/' + str(n_cluster)
    columns, preprocessed_df = prepare_data(df)

    max_acc = 0.5
    for trial in clusters.columns:
        print(trial)
        y = clusters[trial]
        stratify_on = stratify_check(y)

        # Split into train/test
        if stratify_on:
            x_train, x_test, y_train, y_test = train_test_split(preprocessed_df, y, test_size=0.2,
                                                                random_state=1,
                                                                stratify=y)
        else:
            x_train, x_test, y_train, y_test = train_test_split(preprocessed_df, y, test_size=0.2,
                                                                random_state=1)
    model, y_pred, max_acc = classify(x_train, y_train, x_test, y_test,
                                          directory=directory, n_cluster=n_cluster,
                                          trial_no=trial, max_score=max_acc)
    generate_plots(directory, columns, model, n_cluster, x_test, x_train, y_pred, y_test)


if __name__ == '__main__':
    n_clusters = [2, 3, 4, 5, 6, 7]
    for n in n_clusters:
        clf_stability_analysis(n)
