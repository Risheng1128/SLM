import sys
import os
sys.path.append(os.getcwd())

from sklearn import svm
from melt_train_model import dataset
from melt_constants import tensile_key, pmb_key, iron_key
from melt_constants import bone_fp, bone_ppfp, ring_fp, ring_ppfp

svr_param = {'C': [10, 100, 1000],
             'kernel': ['rbf'],
             'gamma': ['auto']}
retain_feature = [13, 12, 11, 10, 9, 8]
layer = 70

for feature in retain_feature:
    tensile_data_set = dataset(tensile_key, output=None)
    tensile_data_set.load_data(bone_fp, bone_ppfp)
    tensile_data_set.reshape_and_repeat((-1, 13), repeat=layer)
    tensile_data_set.mutual_information(feature)
    tensile_data_set.grid_search(svm.SVR(), svr_param, standard_scaler=True)

    pmb_data_set = dataset(pmb_key, output=None)
    pmb_data_set.load_data(ring_fp, ring_ppfp)
    pmb_data_set.reshape_and_repeat((-1, 13), repeat=layer)
    pmb_data_set.mutual_information(feature)
    pmb_data_set.grid_search(svm.SVR(), svr_param, standard_scaler=True)

    iron_data_set = dataset(iron_key, output=None)
    iron_data_set.load_data(ring_fp, ring_ppfp)
    iron_data_set.reshape_and_repeat((-1, 13), repeat=layer)
    iron_data_set.mutual_information(feature)
    iron_data_set.grid_search(svm.SVR(), svr_param, standard_scaler=True)
