import sys
import os
import xgboost as xgb
sys.path.append(os.getcwd())

from melt_train_model import dataset
from melt_constants import tensile_key, pmb_key, iron_key
from melt_constants import bone_filepath, bone_property_filepath, \
    ring_filepath, ring_property_filepath

xgboost_parameter = {'learning_rate': [0.1, 0.15, 0.2, 0.25, 0.3,
                                       0.35, 0.4, 0.45, 0.5],
                     'max_depth': [1, 2, 3, 4, 5,
                                   6, 7, 8, 9, 10],
                     'n_estimators': [100, 500]}
retain_feature = [13, 12, 11, 10, 9, 8]
layer = 70

for feature in retain_feature:
    tensile_data_set = dataset(tensile_key, output=None)
    tensile_data_set.load_data(bone_filepath, bone_property_filepath)
    tensile_data_set.reshape_and_repeat((-1, 13), repeat=layer)
    tensile_data_set.mutual_information(feature)
    tensile_data_set.random_search(xgb.XGBRegressor(), xgboost_parameter)

    pmb_data_set = dataset(pmb_key, output=None)
    pmb_data_set.load_data(ring_filepath, ring_property_filepath)
    pmb_data_set.reshape_and_repeat((-1, 13), repeat=layer)
    pmb_data_set.mutual_information(feature)
    pmb_data_set.random_search(xgb.XGBRegressor(), xgboost_parameter)

    iron_data_set = dataset(iron_key, output=None)
    iron_data_set.load_data(ring_filepath, ring_property_filepath)
    iron_data_set.reshape_and_repeat((-1, 13), repeat=layer)
    iron_data_set.mutual_information(feature)
    iron_data_set.random_search(xgb.XGBRegressor(), xgboost_parameter)
