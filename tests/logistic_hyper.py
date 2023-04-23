import sys
import os
sys.path.append(os.getcwd())

from sklearn.linear_model import LogisticRegression
from melt_train_model import dataset
from melt_constants import tensile_key, pmb_key, iron_key
from melt_constants import bone_train_fp, bone_test_fp
from melt_constants import ring_train_fp, ring_test_fp

logistic_param = {'max_iter': [10000, 20000, 30000],
                  'random_state': [0]}
retain_feature = [18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5]
layer = 70

for feature in retain_feature:
    tensile_data_set = dataset(tensile_key, output=None)
    tensile_data_set.load_train_data(bone_train_fp, [], True)
    tensile_data_set.load_test_data(bone_test_fp, [], True)
    tensile_data_set.reshape_and_repeat((-1, 18), repeat=layer)
    tensile_data_set.mutual_information(feature)
    tensile_data_set.grid_search(LogisticRegression(), logistic_param,
                                 standard_scaler=True, encode=True)

    pmb_data_set = dataset(pmb_key, output=None)
    pmb_data_set.load_train_data(ring_train_fp, [], True)
    pmb_data_set.load_test_data(ring_test_fp, [], True)
    pmb_data_set.reshape_and_repeat((-1, 18), repeat=layer)
    pmb_data_set.mutual_information(feature)
    pmb_data_set.grid_search(LogisticRegression(), logistic_param,
                             standard_scaler=True, encode=True)

    iron_data_set = dataset(iron_key, output=None)
    iron_data_set.load_train_data(ring_train_fp, [], True)
    iron_data_set.load_test_data(ring_test_fp, [], True)
    iron_data_set.reshape_and_repeat((-1, 18), repeat=layer)
    iron_data_set.mutual_information(feature)
    iron_data_set.grid_search(LogisticRegression(), logistic_param,
                              standard_scaler=True, encode=True)
