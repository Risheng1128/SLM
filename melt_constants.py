# excel information
layer_label = ['layer']
feature = ['energy', 'entropy', 'contrast', 'idm', 'autocorrelation', 'mean_x',
           'mean_y', 'variance_x', 'variance_y', 'standard_deviation_x',
           'standard_deviation_y', 'correlation', 'dissimilarity']
proc_param = ['oxygen concentration', 'laser power', 'scanning velocity',
              'layer height', 'energy density']
train_sheet = 'train'
test_sheet = 'test'
ignore_data = 'X'

# output excel information
output_label = ['prediction', 'true', 'error(%)', 'train number',
                'test number', 'feature numebr', 'remove feature',
                'R2 score', 'MSE', 'MAE']

# model default parameters
xgboost_param = {'n_estimators': 1000,
                 'learning_rate': 0.3,
                 'max_depth': 5}
lightgbm_param = {'boosting_type': 'gbdt',
                  'num_leaves': 1000,
                  'learning_rate': 0.3,
                  'max_depth': 5}
logistic_param = {'max_iter': 10000,
                  'random_state': 0}
svr_param = {'C': 1000,
             'kernel': 'rbf',
             'gamma': 'auto'}

# default train and test data
bone_train_fp = ['./data/glcm-data/first_bone_train.xlsx',
                 './data/glcm-data/second_bone_train.xlsx']
bone_test_fp = ['./data/glcm-data/first_bone_test.xlsx',
                './data/glcm-data/second_bone_test.xlsx']
ring_train_fp = ['./data/glcm-data/first_ring_train.xlsx',
                 './data/glcm-data/second_ring_train.xlsx']
ring_test_fp = ['./data/glcm-data/first_ring_test.xlsx',
                './data/glcm-data/second_ring_test.xlsx']

# dictionary key
tensile_key = ['tensile']
pmb_key = ['pmb_50Hz', 'pmb_200Hz', 'pmb_400Hz', 'pmb_800Hz']
iron_key = ['iron_50Hz', 'iron_200Hz', 'iron_400Hz', 'iron_800Hz']
