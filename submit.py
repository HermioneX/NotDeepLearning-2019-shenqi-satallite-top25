# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 09:59:18 2019

@author: HermoineX_zhanling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from parameters import *
import pickle


def get_train_test(train_dataset):
    x, y = train_dataset.iloc[:,1:].values, train_dataset.iloc[:,0].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
    return x_train, x_test, y_train, y_test

def mae_count(ans, y_test):
    return np.mean(abs(ans - y_test))

def xgb_train(x_train, x_test, y_train):
    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=False, objective='reg:gamma')
    model.fit(x_train, y_train)
    ans_xgb = model.predict(x_test)
    return ans_xgb


if __name__ == '__main__':

    versionNum = '1030'

    mae_score = []
    submit_ans = []

    model_path = './models/' + versionNum + '/'
    train_data_path = sate_path + versionNum + '/train_sate/'
    test_data_path = sate_path + versionNum + '/test_sate/'
    fn_tail = '_sate.csv'

    for station_name in py_name:
        print(station_name)
        if station_name == 'dazi':
            y_submit = np.array((dazi_test*12))/st_min_max_dic['TRI'][1]
            submit_ans.append(y_submit)
        else:
            fn = train_data_path + station_name + fn_tail
            train_dataset = pd.read_csv(fn)
            train_dataset = train_dataset.fillna(train_dataset.mean())  # 缺失值填充

            hour_data = train_dataset[train_dataset['TRI'] > 5].hour.values
            st_t = hour_data[hour_data < 12].min() #日出时间
            ed_t = hour_data[hour_data > 15].max() # 日落时间

        ############################## submit answer ##################################
            fn = test_data_path + station_name + fn_tail
            test_dataset = pd.read_csv(fn)
            # test_dataset = test_dataset.drop(['date'],axis=1)
            test_dataset = test_dataset.fillna(test_dataset.mean())  # 缺失值填充

            for col in test_dataset.columns:
                X = test_dataset[col].values
        #        X_max = X.max(axis=0)
                X_min = st_min_max_dic[col][0]
                X_max = st_min_max_dic[col][1]
                X[X < X_min] = X_min
                X[X > X_max] = X_max
                X1 = (X - X_min)/(X_max-X_min + 1e-7)
                test_dataset[col] = X1

            X_submit = test_dataset.values

            t_submit = []
            for cv_step in range(0,4):
                print(cv_step)

                # rg_rf = pickle.load(open(model_path + 'rg_rf_'+ station_name +'.pickle', "rb"))
                # rg_xgb = pickle.load(open(model_path + 'rg_xgb_'+ station_name +'.pickle', "rb"))
                rg_lgb = pickle.load(open(model_path + 'rg_lgb_'+ station_name + str(cv_step) +'.pickle', "rb"))

                # y1 = rg_rf.predict(X_submit)
                # y_submit = rg_xgb.predict(X_submit)
                y_submit = rg_lgb.predict(X_submit)

                # y_submit = np.mean([y1, y2, y3],0)

                y_submit[X_submit[:,0] < st_t/23.0] = 0
                y_submit[X_submit[:,0] > ed_t/23.0] = 0

                t_submit.append(y_submit)

            submit_ans.append(np.mean(t_submit,0))

    a = np.array(submit_ans).reshape(-1)
    a[a<0] = 0
    a[a>st_min_max_dic['TRI'][1]] = st_min_max_dic['TRI'][1]
    a = np.array((a*st_min_max_dic['TRI'][1]),np.uint16)
    sub_fn = src_path + 'Submit_x.csv'
    sub_df = pd.read_csv(sub_fn)
    sub_df['Y'] = a.copy()
    sub_df.to_csv('./submit_data/submit_' + versionNum + '_lgb.csv',index=None)

    #
