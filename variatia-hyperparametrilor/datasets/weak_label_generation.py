from pyod.utils.utility import standardizer
from pyod.utils.utility import precision_n_scores
from sklearn.metrics import average_precision_score, roc_auc_score
import time
from copy import deepcopy

import os
import numpy as np
import pandas as pd

from utility import get_diff, argmaxatn, flatten

# file_name = '1_ALOI'
# file_name = '27_mnist'
# file_name = '12_SpamBase'
# file_name = 'agnews_0'
# file_name = 'SVHN_0'
# file_name = 'imdb'
file_name = 'yelp'

# data = np.load(os.path.join('clean_data', file_name + '.npz'))
# clean_X = np.asarray(data['X']).astype(float)
# clean_X_norm = standardizer(clean_X)
# clean_y = np.asarray(data['y']).astype(int)
# np.savetxt(os.path.join('clean_data', file_name + '_X.csv'), clean_X,
#            delimiter=",")
# np.savetxt(os.path.join('clean_data', file_name + '_y.csv'), clean_y,
#            delimiter=",")

clean_X = pd.read_csv(os.path.join('clean_data', file_name + '_X.csv'),
                      header=None).to_numpy().astype(float)
clean_X_norm = standardizer(clean_X)
clean_y = pd.read_csv(os.path.join('clean_data', file_name + '_y.csv'),
                      header=None).to_numpy().astype(int)

n_samples = clean_X.shape[0]
assert (n_samples == len(clean_y))

# %% 1 simple rule is to flip labels randomly

noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
rule_indexs = [0, 1, 2, 3]

os.makedirs(os.path.join("noisy_data", "flipping", file_name), exist_ok=True)

for rule_index in rule_indexs:
    for noise_level in noise_levels:
        n_noise_samples = int(n_samples * noise_level)
        random_state = np.random.RandomState(rule_index)
        flipped_idx = random_state.choice(len(clean_y), n_noise_samples)
        noisy_y = deepcopy(clean_y)
        noisy_y[flipped_idx] ^= 1

        # check length
        assert (len(noisy_y) == len(clean_y))

        # check if it is binary
        assert (np.unique(noisy_y).tolist() == [0, 1])

        # check quality
        print(roc_auc_score(clean_y, noisy_y))
        assert (roc_auc_score(clean_y, noisy_y) > 0.5)

        np.savetxt(os.path.join("noisy_data", "flipping", file_name,
                                file_name + "_y_" + str(
                                    noise_level) + '_v' + str(
                                    rule_index) + ".csv"), noisy_y,
                   delimiter=",")

# %% 2 learn classifier to generate weak-supervisions
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

supervision_levels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
# supervision_levels = [0.1]
classifiers = [
    MLPClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LGBMClassifier(),
]

os.makedirs(os.path.join("noisy_data", "classification", file_name),
            exist_ok=True)

for supervision_level in supervision_levels:

    n_clean_samples = int(n_samples * supervision_level)
    random_state = np.random.RandomState(42)
    clean_idx = random_state.choice(len(clean_y), n_clean_samples)
    X_train = clean_X_norm[clean_idx, :]
    y_train = clean_y[clean_idx].ravel()

    for clf in classifiers:
        clf_name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        noisy_y = clf.predict(clean_X_norm)

        # check length
        assert (len(noisy_y) == len(clean_y))

        # check if it is binary
        assert (np.unique(noisy_y).tolist() == [0, 1])

        # check quality
        print(roc_auc_score(clean_y, noisy_y))
        assert (roc_auc_score(clean_y, noisy_y) > 0.5)

        np.savetxt(os.path.join("noisy_data", "classification", file_name,
                                file_name + "_y_" + str(
                                    supervision_level) + '_' + str(
                                    clf_name) + ".csv"), noisy_y,
                   delimiter=",")
