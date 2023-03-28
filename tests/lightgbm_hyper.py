import sys
import os
import lightgbm as gbm
sys.path.append(os.getcwd())

from melt_train_model import dataset
from melt_constants import tensile_key, pmb_key, iron_key
from melt_constants import bone_fp, bone_ppfp, ring_fp, ring_ppfp

lightgbm_param = {'boosting_type': ['gbdt', 'dart', 'goss'],
                  'learning_rate': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5],
                  'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  'num_leaves': [100]}
retain_feature = [13, 12, 11, 10, 9, 8]
layer = 70

for feature in retain_feature:
    tensile_data_set = dataset(tensile_key, output=None)
    tensile_data_set.load_data(bone_fp, bone_ppfp)
    tensile_data_set.reshape_and_repeat((-1, 13), repeat=layer)
    tensile_data_set.mutual_information(feature)
    tensile_data_set.random_search(gbm.LGBMRegressor(), lightgbm_param)

    pmb_data_set = dataset(pmb_key, output=None)
    pmb_data_set.load_data(ring_fp, ring_ppfp)
    pmb_data_set.reshape_and_repeat((-1, 13), repeat=layer)
    pmb_data_set.mutual_information(feature)
    pmb_data_set.random_search(gbm.LGBMRegressor(), lightgbm_param)

    iron_data_set = dataset(iron_key, output=None)
    iron_data_set.load_data(ring_fp, ring_ppfp)
    iron_data_set.reshape_and_repeat((-1, 13), repeat=layer)
    iron_data_set.mutual_information(feature)
    iron_data_set.random_search(gbm.LGBMRegressor(), lightgbm_param)
