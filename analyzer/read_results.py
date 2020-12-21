import joblib
import pandas as pd
import numpy as np
import pickle

# with open("../training_data/results_gridsearch_intra_random500.pkl", 'rb') as res:
#     scores = pickle.load(res)
scores_df = pd.read_pickle('results_unsupervised.pkl')
print(scores_df)