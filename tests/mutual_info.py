import sys
import os
sys.path.append(os.getcwd())

from train_model import dataset
from constant import tensile_key, pmb_key, iron_key
from constant import bone_train_fp, bone_test_fp
from constant import ring_train_fp, ring_test_fp
from constant import feature, proc_param

feature_num = 18
layer = 70

tensile_data_set = dataset(tensile_key, output=None)
tensile_data_set.load_train_data(bone_train_fp, [], True)
tensile_data_set.load_test_data(bone_test_fp, [], True)
tensile_data_set.reshape_and_repeat((-1, 18), repeat=layer)
mi = tensile_data_set.mutual_information(feature_num)
for key in tensile_key:
    print('---------------', key, '---------------')
    for i, j in zip(feature + proc_param, mi.read(key)):
        print(i, j)

pmb_data_set = dataset(pmb_key, output=None)
pmb_data_set.load_train_data(ring_train_fp, [], True)
pmb_data_set.load_test_data(ring_test_fp, [], True)
pmb_data_set.reshape_and_repeat((-1, 18), repeat=layer)
mi = pmb_data_set.mutual_information(feature_num)
for key in pmb_key:
    print('---------------', key, '---------------')
    for i, j in zip(feature + proc_param, mi.read(key)):
        print(i, j)

iron_data_set = dataset(iron_key, output=None)
iron_data_set.load_train_data(ring_train_fp, [], True)
iron_data_set.load_test_data(ring_test_fp, [], True)
iron_data_set.reshape_and_repeat((-1, 18), repeat=layer)
mi = iron_data_set.mutual_information(feature_num)
for key in iron_key:
    print('---------------', key, '---------------')
    for i, j in zip(feature + proc_param, mi.read(key)):
        print(i, j)
