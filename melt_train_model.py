import xgboost as xgb
import lightgbm as gbm
import pandas as pd
import numpy as np
import openpyxl
import argparse
import pickle
import os
import melt_constants as const

from sklearn import metrics
from sklearn import svm
from sklearn.preprocessing import StandardScaler

class data:
    def __init__(self, keys):
        self.data = {key: [] for key in keys}

    def wp_append(self, wp_data, key, remove_header):
        self.data[key].append(wp_data.drop(remove_header, axis=1))

    def pp_append(self, pp_data, key):
        self.data[key].append(pp_data)

    def reshape(self, key, shape):
        self.data[key] = np.reshape(np.array(self.data[key]), shape)

    def repeat(self, key, repeat):
        self.data[key] = np.repeat(self.data[key], repeat)

    def get_key_data(self, key):
        return self.data[key]

class dataset:
    def __init__(self, keys, output):
        self.x_train = data(keys)
        self.x_test = data(keys)
        self.y_train = data(keys)
        self.y_test = data(keys)
        self.keys = keys
        self.output = output

    # load workpiece and material property data from excel file
    def load_data(self, workpiece_filepath, property_filepath, header=[]):
        remove_header = const.layer_header + header
        for wpf, ppf in zip(workpiece_filepath, property_filepath):
            print('read workpiece file: ', wpf)
            print('read material property file: ', ppf)
            wp_sheets = pd.read_excel(wpf, sheet_name=None)

            for key in self.keys:
                # the dictionary key must same in property excel
                pp_data = pd.read_excel(ppf, sheet_name=key)
                pp_data = np.array(pp_data.drop(const.trail_header, axis=1))
                # switch 2D array to 1D array
                pp_data = np.reshape(pp_data, -1)

                for index, sheet in zip(range(len(pp_data)), wp_sheets):
                    wp_data = pd.read_excel(wpf, sheet_name=sheet)
                    if pp_data[index] == const.error_data:
                        continue

                    if sheet[-1] == '5' or sheet[-1] == '6':
                        self.x_test.wp_append(wp_data, key, remove_header)
                        self.y_test.pp_append(pp_data[index], key)
                    else:
                        self.x_train.wp_append(wp_data, key, remove_header)
                        self.y_train.pp_append(pp_data[index], key)

    # change the data shape or repeat same data
    def reshape_and_repeat(self, shape, repeat=1):
        for key in self.keys:
            # convert 3-D matrix to 2-D matrix
            self.x_train.reshape(key, shape)
            self.x_test.reshape(key, shape)
            self.y_train.repeat(key, repeat)
            self.y_test.repeat(key, repeat)

    # show the correlation coefficient
    def show_corrcoef(self):
        for key in self.keys:
            x_train, x_test, y_train, y_test = self.get_key_data(key)
            x = np.vstack((x_train, x_test))
            y = np.concatenate((y_train, y_test))
            x_mean = np.mean(x, axis=0)
            y_mean = np.mean(y)

            numerator = np.zeros(shape=x_mean.shape[0])
            denominator1 = np.zeros(shape=x_mean.shape[0])
            denominator2 = np.zeros(shape=x_mean.shape[0])

            # compute correlation coefficient
            for xi, yi in zip(x, y):
                x_tmp = xi - x_mean
                y_tmp = yi - y_mean
                numerator += x_tmp * y_tmp
                denominator1 += x_tmp ** 2
                denominator2 += y_tmp ** 2

            r = numerator / (np.sqrt(denominator1) * np.sqrt(denominator2))
            print('----------', key, '----------')
            for feature, i in zip(const.feature_header, r):
                print(feature, ': ', i)

    def get_key_data(self, key):
        return self.x_train.get_key_data(key), self.x_test.get_key_data(key), \
            self.y_train.get_key_data(key), self.y_test.get_key_data(key)

    # store data into excel
    def store_data(self, key, sheet, predict):
        x_train, x_test, _, y_test = self.get_key_data(key)

        for col, header in zip(range(1, len(const.output_header) + 1),
                               const.output_header):
            sheet.cell(1, col).value = header
        sheet.cell(2, 4).value = x_train.shape[0]
        sheet.cell(2, 5).value = x_test.shape[0]
        # compute R2 score, MSE and MAE
        sheet.cell(2, 6).value = metrics.r2_score(y_test, predict)
        sheet.cell(2, 7).value = metrics.mean_squared_error(y_test, predict)
        sheet.cell(2, 8).value = metrics.mean_absolute_error(y_test, predict)

        row = 2
        for pre, true in zip(predict, y_test):
            sheet.cell(row, 1).value = pre
            sheet.cell(row, 2).value = true
            sheet.cell(row, 3).value = (abs(pre - true) / true) * 100
            row += 1

    # create sheet and store data into excel
    def create_sheet_and_store_data(self, key, wb, predict, xlsx):
        # create new sheet
        sheet = wb.create_sheet(key)
        # store data into excel
        self.store_data(key, sheet, predict)
        # save the excel
        wb.save(self.output + xlsx)

    def train_xgboost_model(self, xlsx):
        wb = openpyxl.Workbook()
        for key in self.keys:
            x_train, x_test, y_train, _ = self.get_key_data(key)
            model = xgb.XGBRegressor(n_estimators=10000,
                                     learning_rate=0.3,
                                     max_depth=5)
            # train XGBoost model
            model.fit(x_train, y_train)
            # save XGBoost model
            model_name = self.output + key + '_xgboost.pickle.dat'
            pickle.dump(model, open(model_name, 'wb'))
            # predict x_test via XGBoost model
            predict = model.predict(x_test)
            # create sheet and store data into excel
            self.create_sheet_and_store_data(key, wb, predict, xlsx)

    def train_lightgbm_model(self, xlsx):
        wb = openpyxl.Workbook()
        for key in self.keys:
            x_train, x_test, y_train, _ = self.get_key_data(key)
            model = gbm.LGBMRegressor(boosting_type='gbdt', num_leaves=10000,
                                      learning_rate=0.3, max_depth=6)
            # train lightGBM model
            model.fit(x_train, y_train)
            # save lightGBM model
            model_name = self.output + key + '_lightgbm.pickle.dat'
            pickle.dump(model, open(model_name, 'wb'))
            # predict x_test via lightGBM model
            predict = model.predict(x_test)
            # create sheet and store data into excel
            self.create_sheet_and_store_data(key, wb, predict, xlsx)

    def train_svr_model(self, xlsx):
        wb = openpyxl.Workbook()
        for key in self.keys:
            x_train, x_test, y_train, _ = self.get_key_data(key)
            # normalzed
            ss = StandardScaler()
            x_train = ss.fit_transform(x_train)
            x_test = ss.fit_transform(x_test)

            model = svm.SVR(C=1000, kernel='rbf', gamma='auto')
            # train support vector regression model
            model.fit(x_train, y_train)
            # save lightGBM model
            model_name = self.output + key + '_svr.pickle.dat'
            pickle.dump(model, open(model_name, 'wb'))
            # predict x_test via lightGBM model
            predict = model.predict(x_test)
            # create sheet and store data into excel
            self.create_sheet_and_store_data(key, wb, predict, xlsx)

    def display_all_data(self):
        for key in self.keys:
            print('-----------', key, '-----------')
            print('x_train = ', self.x_train.get_key_data(key))
            print('x_test = ', self.x_test.get_key_data(key))
            print('y_train = ', self.y_train.get_key_data(key))
            print('y_test = ', self.y_test.get_key_data(key))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dst',
                        default='./result/xgboost/',
                        help='destination path')
    args = parser.parse_args()

    if not os.path.isdir(args.dst):
        os.makedirs(args.dst)

    bone_filepath = ['./data/glcm-data/第一批狗骨頭.xlsx',
                     './data/glcm-data/第二批狗骨頭.xlsx']

    bone_property_filepath = ['./data/glcm-data/第一批狗骨頭材料特性.xlsx',
                              './data/glcm-data/第二批狗骨頭材料特性.xlsx']

    ring_filepath = ['./data/glcm-data/第一批圓環.xlsx',
                     './data/glcm-data/第二批圓環.xlsx']

    ring_property_filepath = ['./data/glcm-data/第一批圓環材料特性.xlsx',
                              './data/glcm-data/第二批圓環材料特性.xlsx']

    # train tensile model
    tensile_data_set = dataset(const.tensile_key, output=args.dst)
    tensile_data_set.load_data(bone_filepath, bone_property_filepath)
    tensile_data_set.reshape_and_repeat((-1, 70 * 13), repeat=1)
    tensile_data_set.train_xgboost_model(xlsx='tensile_xgboost.xlsx')
    tensile_data_set.train_lightgbm_model(xlsx='tensile_lightgbm.xlsx')
    tensile_data_set.train_svr_model(xlsx='tensile_svr.xlsx')

    # train permeability model
    pmb_data_set = dataset(const.pmb_key, output=args.dst)
    pmb_data_set.load_data(ring_filepath, ring_property_filepath)
    pmb_data_set.reshape_and_repeat((-1, 13), repeat=70)
    pmb_data_set.train_xgboost_model(xlsx='pmb_xgboost.xlsx')
    pmb_data_set.train_lightgbm_model(xlsx='pmb_lightgbm.xlsx')
    pmb_data_set.train_svr_model(xlsx='pmb_svr.xlsx')

    # train iron loss model
    iron_data_set = dataset(const.iron_key, output=args.dst)
    iron_data_set.load_data(ring_filepath, ring_property_filepath)
    iron_data_set.reshape_and_repeat((-1, 70 * 13), repeat=1)
    iron_data_set.train_xgboost_model(xlsx='iron_xgboost.xlsx')
    iron_data_set.train_lightgbm_model(xlsx='iron_lightgbm.xlsx')
    iron_data_set.train_svr_model(xlsx='iron_svr.xlsx')
