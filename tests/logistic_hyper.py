import sys
import os
sys.path.append(os.getcwd())

from sklearn.linear_model import LogisticRegression
from melt_train_model import dataset
from melt_constants import tensile_key, pmb_key, iron_key
from melt_constants import bone_fp, bone_ppfp, ring_fp, ring_ppfp

logistic_param = {'max_iter': [10000, 20000, 30000],
                  'random_state': [0]}
retain_feature = [13, 12, 11, 10, 9, 8]
layer = 70

for feature in retain_feature:
    tensile_data_set = dataset(tensile_key, output=None)
    tensile_data_set.load_data(bone_fp, bone_ppfp)
    tensile_data_set.reshape_and_repeat((-1, 13), repeat=layer)
    tensile_data_set.mutual_information(feature)
    tensile_data_set.grid_search(LogisticRegression(), logistic_param,
                                 standard_scaler=True, encode=True)

    pmb_data_set = dataset(pmb_key, output=None)
    pmb_data_set.load_data(ring_fp, ring_ppfp)
    pmb_data_set.reshape_and_repeat((-1, 13), repeat=layer)
    pmb_data_set.mutual_information(feature)
    pmb_data_set.grid_search(LogisticRegression(), logistic_param,
                             standard_scaler=True, encode=True)

    iron_data_set = dataset(iron_key, output=None)
    iron_data_set.load_data(ring_fp, ring_ppfp)
    iron_data_set.reshape_and_repeat((-1, 13), repeat=layer)
    iron_data_set.mutual_information(feature)
    iron_data_set.grid_search(LogisticRegression(), logistic_param,
                              standard_scaler=True, encode=True)
