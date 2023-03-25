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
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV

class data:
    def __init__(self, keys):
        self.__data = {key: [] for key in keys}

    def wp_append(self, wp_data, key, remove_header):
        self.__data[key].append(wp_data.drop(remove_header, axis=1))

    def pp_append(self, pp_data, key):
        self.__data[key].append(pp_data)

    def reshape(self, key, shape):
        self.__data[key] = np.reshape(np.array(self.__data[key]), shape)

    def repeat(self, key, repeat):
        self.__data[key] = np.repeat(self.__data[key], repeat)

    def unique(self, key, k):
        self.__data[key] = self.__data[key][::k]

    def read_key_data(self, key):
        return self.__data[key]

    def write_key_data(self, key, data):
        self.__data[key] = data

class xgboost:
    def __init__(self):
        self.__n_estimators = 1000
        self.__learning_rate = 0.3
        self.__max_depth = 5

    def write(self, n_estimators=1000, learning_rate=0.3, max_depth=5):
        self.__n_estimators = n_estimators
        self.__learning_rate = learning_rate
        self.__max_depth = max_depth

    def read(self):
        return self.__n_estimators, self.__learning_rate, self.__max_depth

class lightgbm:
    def __init__(self):
        self.__boosting_type = 'gbdt'
        self.__num_leaves = 1000
        self.__learning_rate = 0.3
        self.__max_depth = 5

    def write(self, boosting_type='gbdt', num_leaves=1000, learning_rate=0.3,
              max_depth=5):
        self.__boosting_type = boosting_type
        self.__num_leaves = num_leaves
        self.__learning_rate = learning_rate
        self.__max_depth = max_depth

    def read(self):
        return self.__boosting_type, self.__num_leaves, \
            self.__learning_rate, self.__max_depth

class logistic_regression:
    def __init__(self):
        self.__max_iter = 10000
        self.__random_state = 0

    def write(self, max_iter=10000, random_state=0):
        self.__max_iter = max_iter
        self.__random_state = random_state

    def read(self):
        return self.__max_iter, self.__random_state

class svr:
    def __init__(self):
        self.__C = 1000
        self.__kernel = 'rbf'
        self.__gamma = 'auto'

    def write(self, C=1000, kernel='rbf', gamma='auto'):
        self.__C = C
        self.__kernel = kernel
        self.__gamma = gamma

    def read(self):
        return self.__C, self.__kernel, self.__gamma

class dataset:
    def __init__(self, keys, output, feature_num=13):
        self.__x_train = data(keys)
        self.__x_test = data(keys)
        self.__y_train = data(keys)
        self.__y_test = data(keys)
        self.__xgboost = xgboost()
        self.__lightgbm = lightgbm()
        self.__logistic = logistic_regression()
        self.__svr = svr()
        self.__keys = keys
        self.__output = output
        self.__feature_num = {key: feature_num for key in keys}
        self.__remove_feature = {key: [] for key in keys}
        self.__grid_search_parameter

    def __read_key_data(self, key):
        return self.__x_train.read_key_data(key), \
            self.__x_test.read_key_data(key), \
            self.__y_train.read_key_data(key), \
            self.__y_test.read_key_data(key)

    # do standard scaler tp x_train and x_test
    def __standard_scaler(self, x_train, x_test):
        ss = StandardScaler()
        x_train = ss.fit_transform(x_train)
        x_test = ss.fit_transform(x_test)
        return x_train, x_test

    # store data in excel (row direction)
    def __store_data(self, sheet, datas, base_row, base_col):
        for col, data in zip(range(base_col, base_col + len(datas)), datas):
            sheet.cell(base_row, col).value = data

    # store model result into excel
    def __store_result_data(self, key, sheet, predict):
        x_train, x_test, _, y_test = self.__read_key_data(key)
        datas = [x_train.shape[0],
                 x_test.shape[0],
                 self.__feature_num[key],
                 ', '.join(map(str, self.__remove_feature[key])),
                 metrics.r2_score(y_test, predict),
                 metrics.mean_squared_error(y_test, predict),
                 metrics.mean_absolute_error(y_test, predict)]

        self.__store_data(sheet, const.output_header, 1, 1)
        self.__store_data(sheet, datas, 2, 4)

        row = 2
        for pre, true in zip(predict, y_test):
            sheet.cell(row, 1).value = pre
            sheet.cell(row, 2).value = true
            sheet.cell(row, 3).value = (abs(pre - true) / true) * 100
            row += 1

    # store XGBoost model setting into excel
    def __store_xgboost_setting(self, sheet):
        n_estimators, learning_rate, max_depth = self.__xgboost.read()
        datas = [n_estimators, learning_rate, max_depth]

        self.__store_data(sheet, const.xgboost_header, 3, 4)
        self.__store_data(sheet, datas, 4, 4)

    # store lightGBM model setting into excel
    def __store_lightgbm_setting(self, sheet):
        boosting_type, num_leaves, learning_rate, max_depth = \
            self.__lightgbm.read()
        datas = [boosting_type, num_leaves, learning_rate, max_depth]

        self.__store_data(sheet, const.lightgbm_header, 3, 4)
        self.__store_data(sheet, datas, 4, 4)

    # store logistic regression model setting into excel
    def __store_logistic_setting(self, sheet):
        max_iter, random_state = self.__logistic.read()
        datas = [max_iter, random_state]

        self.__store_data(sheet, const.logistic_header, 3, 4)
        self.__store_data(sheet, datas, 4, 4)

    # store SVR model setting into excel
    def __store_svr_setting(self, sheet):
        C, kernel, gamma = self.__svr.read()
        datas = [C, kernel, gamma]

        self.__store_data(sheet, const.svr_header, 3, 4)
        self.__store_data(sheet, datas, 4, 4)

    # load workpiece and material property data from excel
    def load_data(self, workpiece_filepath, property_filepath, header=[]):
        remove_header = const.layer_header + header
        for wpf, ppf in zip(workpiece_filepath, property_filepath):
            print('read workpiece file: ', wpf)
            print('read material property file: ', ppf)
            wp_sheets = pd.read_excel(wpf, sheet_name=None)

            for key in self.__keys:
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
                        self.__x_test.wp_append(wp_data, key, remove_header)
                        self.__y_test.pp_append(pp_data[index], key)
                    else:
                        self.__x_train.wp_append(wp_data, key, remove_header)
                        self.__y_train.pp_append(pp_data[index], key)

    # change the data shape or repeat same data
    def reshape_and_repeat(self, shape, repeat=1):
        for key in self.__keys:
            # convert 3-D matrix to 2-D matrix
            self.__x_train.reshape(key, shape)
            self.__x_test.reshape(key, shape)
            self.__y_train.repeat(key, repeat)
            self.__y_test.repeat(key, repeat)

    # get all unique data in repeated array
    def unique(self, k):
        for key in self.__keys:
            self.__y_train.unique(key, k)
            self.__y_test.unique(key, k)

    # show the correlation coefficient
    def show_corrcoef(self):
        for key in self.keys:
            x_train, x_test, y_train, y_test = self.__read_key_data(key)
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

    # compute mutual information and retain "k" best data
    def mutual_information(self, k=3):
        for key in self.__keys:
            x_train, x_test, y_train, _ = self.__read_key_data(key)
            selector = SelectKBest(mutual_info_regression, k=k)
            self.__x_train.write_key_data(key, selector.fit_transform(x_train,
                                                                      y_train))
            # find the index of removing feature
            supports = selector.get_support()
            false_index = np.where(supports == False)[0]
            self.__x_test.write_key_data(key, np.delete(x_test,
                                                        false_index,
                                                        axis=1))

            # record retained feature number
            self.__feature_num[key] = k
            # record removed feature
            for feature, support in zip(const.feature_header, supports):
                if support == False:
                    self.__remove_feature[key].append(feature)

    # principal component analysis
    def PCA(self, components):
        pca = PCA(n_components=components)
        for key in self.__keys:
            pca_x_train = pca.fit_transform(self.__x_train.read_key_data(key))
            pca_x_test = pca.fit_transform(self.__x_test.read_key_data(key))
            self.__x_train.write_key_data(key, pca_x_train)
            self.__x_test.write_key_data(key, pca_x_test)

    # t-distributed stochastic neighbor embedding
    def TSNE(self, components):
        tsne = TSNE(n_components=components)
        for key in self.__keys:
            tsne_x_train = tsne.fit_transform(self.__x_train.read_key_data(key))
            tsne_x_test = tsne.fit_transform(self.__x_test.read_key_data(key))
            self.__x_train.write_key_data(key, tsne_x_train)
            self.__x_test.write_key_data(key, tsne_x_test)

    # set XGBoost model parameter
    def xgboost_set(self, n_estimators=1000, learning_rate=0.3, max_depth=5):
        self.__xgboost.write(n_estimators, learning_rate, max_depth)

    # set lightGBM model parameter
    def lightgbm_set(self, boosting_type='gbdt', num_leaves=1000,
                     learning_rate=0.3, max_depth=5):
        self.__lightgbm.write(boosting_type, num_leaves,
                              learning_rate, max_depth)

    def logistic_set(self, max_iter=10000, random_state=0):
        self.__logistic.write(max_iter=max_iter, random_state=random_state)

    # set SVR model parameter
    def svr_set(self, C=1000, kernel='rbf', gamma='auto'):
        self.__svr.write(C, kernel, gamma)

    def grid_search_set(self, parameter):
        self.__grid_search_parameter = parameter

    def grid_search(self, model):
        for key in self.__keys:
            x_train, _, y_train, _ = self.__read_key_data(key)
            grid_search = GridSearchCV(model, self.__grid_search_parameter)
            grid_search.fit(x_train, y_train)
            print(grid_search.cv_results_)

    def xgboost(self, xlsx):
        wb = openpyxl.Workbook()
        for key in self.__keys:
            x_train, x_test, y_train, _ = self.__read_key_data(key)
            n_estimators, learning_rate, max_depth = self.__xgboost.read()

            model = xgb.XGBRegressor(n_estimators=n_estimators,
                                     learning_rate=learning_rate,
                                     max_depth=max_depth)
            # train XGBoost model
            model.fit(x_train, y_train)
            # predict y_test via XGBoost model
            predict = model.predict(x_test)

            # create new sheet
            sheet = wb.create_sheet(key)
            # store data into excel
            self.__store_result_data(key, sheet, predict)
            self.__store_xgboost_setting(sheet)
            # save the excel
            wb.save(self.__output + xlsx)

            # save XGBoost model
            model_name = self.__output + key + '_xgboost.pickle.dat'
            pickle.dump(model, open(model_name, 'wb'))

    def lightgbm(self, xlsx):
        wb = openpyxl.Workbook()
        for key in self.__keys:
            x_train, x_test, y_train, _ = self.__read_key_data(key)
            boosting_type, num_leaves, learning_rate, max_depth = \
                self.__lightgbm.read()

            model = gbm.LGBMRegressor(boosting_type=boosting_type,
                                      num_leaves=num_leaves,
                                      learning_rate=learning_rate,
                                      max_depth=max_depth)
            # train lightGBM model
            model.fit(x_train, y_train)
            # predict y_test via lightGBM model
            predict = model.predict(x_test)

            # create new sheet
            sheet = wb.create_sheet(key)
            # store data into excel
            self.__store_result_data(key, sheet, predict)
            self.__store_lightgbm_setting(sheet)
            # save the excel
            wb.save(self.__output + xlsx)

            # save lightGBM model
            model_name = self.__output + key + '_lightgbm.pickle.dat'
            pickle.dump(model, open(model_name, 'wb'))

    def linear_regression(self, xlsx):
        wb = openpyxl.Workbook()
        for key in self.__keys:
            x_train, x_test, y_train, _ = self.__read_key_data(key)
            x_train, x_test = self.__standard_scaler(x_train, x_test)

            model = LinearRegression()
            # train linear regression model
            model.fit(x_train, y_train)
            # predict y_test via linear regression model
            predict = model.predict(x_test)

            # create new sheet
            sheet = wb.create_sheet(key)
            # store data into excel
            self.__store_result_data(key, sheet, predict)
            # save the excel
            wb.save(self.__output + xlsx)

            # save linear regression model
            model_name = self.__output + key + '_linear.pickle.dat'
            pickle.dump(model, open(model_name, 'wb'))

    def logistic_regression(self, xlsx):
        wb = openpyxl.Workbook()
        for key in self.__keys:
            x_train, x_test, y_train, _ = self.__read_key_data(key)
            x_train, x_test = self.__standard_scaler(x_train, x_test)
            max_iter, random_state = self.__logistic.read()
            encoder = LabelEncoder()
            y_train = encoder.fit_transform(y_train)

            model = LogisticRegression(max_iter=max_iter,
                                       random_state=random_state)
            # train logistic regression model
            model.fit(x_train, y_train)
            # predict y_test via logistic regression model
            predict = model.predict(x_test)
            predict = encoder.inverse_transform(predict)

            # create new sheet
            sheet = wb.create_sheet(key)
            # store data into excel
            self.__store_result_data(key, sheet, predict)
            self.__store_logistic_setting(sheet)
            # save the excel
            wb.save(self.__output + xlsx)

            # save logistic regression model
            model_name = self.__output + key + '_logistic.pickle.dat'
            pickle.dump(model, open(model_name, 'wb'))

    def svr(self, xlsx):
        wb = openpyxl.Workbook()
        for key in self.__keys:
            x_train, x_test, y_train, _ = self.__read_key_data(key)
            x_train, x_test = self.__standard_scaler(x_train, x_test)
            C, kernel, gamma = self.__svr.read()

            model = svm.SVR(C=C, kernel=kernel, gamma=gamma)
            # train SVR model
            model.fit(x_train, y_train)
            # predict y_test via SVR model
            predict = model.predict(x_test)

            # create new sheet
            sheet = wb.create_sheet(key)
            # store data into excel
            self.__store_result_data(key, sheet, predict)
            self.__store_svr_setting(sheet)
            # save the excel
            wb.save(self.__output + xlsx)

            # save SVR model
            model_name = self.__output + key + '_svr.pickle.dat'
            pickle.dump(model, open(model_name, 'wb'))

    def display_all_data(self):
        for key in self.__keys:
            print('-----------', key, '-----------')
            print('x_train = ', self.__x_train.read_key_data(key))
            print('x_test = ', self.__x_test.read_key_data(key))
            print('y_train = ', self.__y_train.read_key_data(key))
            print('y_test = ', self.__y_test.read_key_data(key))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dst',
                        default='./result/melt-model/',
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
    retain_feature = 10
    layer = 70
    tensile_data_set = dataset(const.tensile_key, output=args.dst)
    tensile_data_set.load_data(bone_filepath, bone_property_filepath)
    tensile_data_set.reshape_and_repeat((-1, 13), repeat=layer)
    tensile_data_set.mutual_information(retain_feature)
    tensile_data_set.xgboost(xlsx='tensile_xgboost.xlsx')
    tensile_data_set.lightgbm(xlsx='tensile_lightgbm.xlsx')
    tensile_data_set.linear_regression(xlsx='tensile_linear.xlsx')
    tensile_data_set.logistic_regression(xlsx='tensile_logistic.xlsx')
    tensile_data_set.svr(xlsx='tensile_svr.xlsx')

    # train permeability model
    pmb_data_set = dataset(const.pmb_key, output=args.dst)
    pmb_data_set.load_data(ring_filepath, ring_property_filepath)
    pmb_data_set.reshape_and_repeat((-1, 13), repeat=layer)
    pmb_data_set.mutual_information(retain_feature)
    pmb_data_set.xgboost(xlsx='pmb_xgboost.xlsx')
    pmb_data_set.lightgbm(xlsx='pmb_lightgbm.xlsx')
    pmb_data_set.linear_regression(xlsx='pmb_linear.xlsx')
    pmb_data_set.logistic_regression(xlsx='pmb_logistic.xlsx')
    pmb_data_set.svr(xlsx='pmb_svr.xlsx')

    # train iron loss model
    iron_data_set = dataset(const.iron_key, output=args.dst)
    iron_data_set.load_data(ring_filepath, ring_property_filepath)
    iron_data_set.reshape_and_repeat((-1, 13), repeat=layer)
    iron_data_set.mutual_information(retain_feature)
    iron_data_set.xgboost(xlsx='iron_xgboost.xlsx')
    iron_data_set.lightgbm(xlsx='iron_lightgbm.xlsx')
    iron_data_set.linear_regression(xlsx='iron_linear.xlsx')
    iron_data_set.logistic_regression(xlsx='iron_logistic.xlsx')
    iron_data_set.svr(xlsx='iron_svr.xlsx')
