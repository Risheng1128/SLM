import sys
import os
import xgboost as xgb
sys.path.append(os.getcwd())

from train_model import dataset
from constant import tensile_key, pmb_key, iron_key
from constant import bone_train_fp, bone_test_fp
from constant import ring_train_fp, ring_test_fp

xgboost_param = {'learning_rate': [0.1, 0.15, 0.2, 0.25, 0.3,
                                   0.35, 0.4, 0.45, 0.5],
                 'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9,
                               10, 11, 12, 13, 14, 15],
                 'n_estimators': [100, 500]}
retain_feature = [18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5]
layer = 70

for feature in retain_feature:
    tensile_data_set = dataset(tensile_key, output=None)
    tensile_data_set.load_train_data(bone_train_fp, [], True)
    tensile_data_set.load_test_data(bone_test_fp, [], True)
    tensile_data_set.reshape_and_repeat((-1, 18), repeat=layer)
    tensile_data_set.mutual_information(feature)
    tensile_data_set.random_search(xgb.XGBRegressor(), xgboost_param)

    pmb_data_set = dataset(pmb_key, output=None)
    pmb_data_set.load_train_data(ring_train_fp, [], True)
    pmb_data_set.load_test_data(ring_test_fp, [], True)
    pmb_data_set.reshape_and_repeat((-1, 18), repeat=layer)
    pmb_data_set.mutual_information(feature)
    pmb_data_set.random_search(xgb.XGBRegressor(), xgboost_param)

    iron_data_set = dataset(iron_key, output=None)
    iron_data_set.load_train_data(ring_train_fp, [], True)
    iron_data_set.load_test_data(ring_test_fp, [], True)
    iron_data_set.reshape_and_repeat((-1, 18), repeat=layer)
    iron_data_set.mutual_information(feature)
    iron_data_set.random_search(xgb.XGBRegressor(), xgboost_param)
