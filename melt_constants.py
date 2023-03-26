# workpiece excel information
layer_header = ['layer']
feature_header = ['energy', 'entropy', 'contrast', 'idm', 'autocorrelation',
                  'mean_x', 'mean_y', 'variance_x', 'variance_y',
                  'standard_deviation_x', 'standard_deviation_y',
                  'correlation', 'dissimilarity']

# material property excel information
trail_header = ['trail']
error_data = 'X'

# output excel information
output_header = ['prediction', 'true', 'error(%)', 'train number',
                 'test number', 'feature numebr', 'remove feature',
                 'R2 score', 'MSE', 'MAE']

# model default parameters
xgboost_parameter = {'n_estimators': 1000,
                     'learning_rate': 0.3,
                     'max_depth': 5}
lightgbm_parameter = {'boosting_type': 'gbdt',
                      'num_leaves': 1000,
                      'learning_rate': 0.3,
                      'max_depth': 5}
logistic_parameter = {'max_iter': 10000,
                      'random_state': 0}
svr_parameter = {'C': 1000,
                 'kernel': 'rbf',
                 'gamma': 'auto'}

# default train and test data
bone_filepath = ['./data/glcm-data/第一批狗骨頭.xlsx',
                 './data/glcm-data/第二批狗骨頭.xlsx']
bone_property_filepath = ['./data/glcm-data/第一批狗骨頭材料特性.xlsx',
                          './data/glcm-data/第二批狗骨頭材料特性.xlsx']
ring_filepath = ['./data/glcm-data/第一批圓環.xlsx',
                 './data/glcm-data/第二批圓環.xlsx']
ring_property_filepath = ['./data/glcm-data/第一批圓環材料特性.xlsx',
                          './data/glcm-data/第二批圓環材料特性.xlsx']

# dictionary key
tensile_key = ['tensile']
pmb_key = ['pmb_50Hz', 'pmb_200Hz', 'pmb_400Hz', 'pmb_800Hz']
iron_key = ['iron_50Hz', 'iron_200Hz', 'iron_400Hz', 'iron_800Hz']
