feature_header = ['energy', 'entropy', 'contrast', 'idm', 'autocorrelation',
                  'mean_x', 'mean_y', 'variance_x', 'variance_y',
                  'standard_deviation_x', 'standard_deviation_y',
                  'correlation', 'dissimilarity']

# excel header information
layer_header = ['layer']
tensile_header = ['拉伸強度']
pmb_header = ['磁導率(50Hz)', '磁導率(200Hz)', '磁導率(400Hz)', '磁導率(800Hz)']
iron_header = ['鐵損(50Hz)', '鐵損(200Hz)', '鐵損(400Hz)', '鐵損(800Hz)']
output_header = ['prediction', 'true', 'error(%)', 'train number',
                 'test number', 'R2 Score', 'MSE', 'MAE']

error_data = 'X'

# dictionary key
tensile_key = ['tensile']
pmb_key = ['pmb_50Hz', 'pmb_200Hz', 'pmb_400Hz', 'pmb_800Hz']
iron_key = ['iron_50Hz', 'iron_200Hz', 'iron_400Hz', 'iron_800Hz']
