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
import lightgbm as lgb
import pickle
import os
import shutil
from sklearn.model_selection import GridSearchCV
from parameters import *


def get_train_test(train_dataset, cv_step , step_hours = 96):
    x, y = train_dataset.iloc[:,1:].values, train_dataset.iloc[:,0].values
    x_train = np.vstack((x[:step_hours*cv_step+1],x[step_hours*(cv_step+1):])).copy()
    x_test = x[step_hours*cv_step:step_hours*(cv_step+1)].copy()
    y_train = np.hstack((y[:step_hours*cv_step+1],y[step_hours*(cv_step+1):])).copy()
    y_test = y[step_hours*cv_step:step_hours*(cv_step+1)].copy()
    return x_train, x_test, y_train, y_test

def mae_count(ans, y_test):
    return np.mean(abs(ans - y_test))

def station_mae_plot(mae_score, out_pic = './op_pics/train_score.jpg'):

    data = mae_score.copy()
    x = np.arange(data.shape[0])
    #x_labels =
    fig,ax = plt.subplots(figsize = (30,15))

    aveg = np.mean(data[:,2])
    trg = 0.04

    l0 = ax.plot(x,data[:,0], c='blue')
    l1 = ax.plot(x,data[:,1], c='red')
    l2 = ax.plot(x,data[:,2], c='green')
    # l3 = ax.plot(x,data[:,3], c='green')
    l3 = ax.plot(x,np.zeros(data.shape[0]) + aveg, c='orange')
    l4 = ax.plot(x,np.zeros(data.shape[0]) + trg, c = 'purple')

    plt.legend(['xgb', 'lgb','all','avg','0.020_target'],fontsize = 20,loc = 'best')
    plt.grid(b=True,
             color='black',
             linestyle='--',
             linewidth=1,
             alpha=0.3,
             axis='x',
             which="major")

    xticks = range(0,data.shape[0],1)
    xlabels = py_name
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels,rotation=40, size = 15)

    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 30,
    }
    plt.xlabel('stations',font2)
    plt.ylabel('mae',font2)

    # plt.savefig('./other_data/submit_' + py_name[i] + '.jpg')
    plt.savefig(out_pic)

def station_line_plot(tri_series, cnt_title, out_pic = './op_pics/predict.jpg'):

    data = tri_series.copy()

    print(data[0].shape[0])

    x = np.arange(data[0].shape[0])
    #x_labels =
    fig,ax = plt.subplots(figsize = (30,15))


    l1 = ax.plot(x,data[1].copy(), c='red')
    l2 = ax.plot(x,data[2].copy(), c='blue')
    l0 = ax.plot(x,data[0].copy(), c='green')
    # l3 = ax.plot(x,data[3].copy(), c='green')
    # plt.axvlines(x[-96],color="blue")#竖线

    plt.legend(['xgb', 'lgb','true'],fontsize = 20,loc = 'best')
    plt.grid(b=True,
             color='black',
             linestyle='--',
             linewidth=1,
             alpha=0.3,
             axis='x',
             which="major")
    plt.title(cnt_title)

    xticks = range(0,data[0].shape[0],1)
    xlabels = xticks
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=40, size = 15)

    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 30,
    }
    plt.xlabel('stations',font2)
    plt.ylabel('TRI',font2)

    # plt.savefig('./other_data/submit_' + py_name[i] + '.jpg')
    plt.savefig(out_pic)

def rf_model(x_train, y_train, x_test, y_test):
    rg_rf = RandomForestRegressor(n_estimators=200, random_state=0)
    rg_rf.fit(x_train, y_train)
    return rg_rf

def xgb_model(x_train, y_train, x_test, y_test):
    other_params = {'n_estimators': 500, 'learning_rate': 0.03, 'max_depth': 20, 'min_child_weight': 30,
                    'gamma': 0.01}
    # other_params = {'n_estimators': 500, 'learning_rate': 0.03, 'max_depth': 20}
    rg_xgb = xgb.XGBRegressor(**other_params)
    rg_xgb.fit(x_train, y_train)
    return rg_xgb

def lgb_model(x_train, y_train, x_test, y_test, tunning_params = {}):

    params_lgb = {
                'boosting_type': 'gbdt',
                'objective': 'regression_l1',
                'metric': 'mae',
                'num_threads':4}
    trn_data = lgb.Dataset(x_train, y_train)
    val_data = lgb.Dataset(x_test, y_test)
    rg_lgb = lgb.train(dict(params_lgb, **tunning_params ),
                        trn_data,
                        num_boost_round = 3000,
                        valid_sets = [trn_data, val_data],
                        verbose_eval = 500,
                        early_stopping_rounds = 300)
    return  rg_lgb

def predict_norm(model, st_t, ed_t, x_test):
    ans = model.predict(x_test)
    ans[ans<0] = 0
    ans[ans>st_min_max_dic['TRI'][1]] = st_min_max_dic['TRI'][1]
    ans[x_test[:,0] < st_t/23.0] = 0
    ans[x_test[:,0] > ed_t/23.0] = 0
    return ans

if __name__ == '__main__':

    mae_score = []
    submit_ans = []

    versionNum = '1030'

    if os.path.exists('./models/' + versionNum):
        shutil.rmtree('./models/' + versionNum)
    # if os.path.exists('./op_pics/predict/'+ versionNum):
    #     shutil.rmtree('./op_pics/predict/'+ versionNum)
    os.mkdir('./models/' + versionNum)
    # os.mkdir('./op_pics/predict/'+ versionNum)

    for station_name in py_name:
    # for station_name in ['yichang']:

    # station_name = 'dazi'

        print(station_name)

        fn = sate_path + versionNum + '/train_sate/' + station_name + '_sate.csv'
        train_dataset = pd.read_csv(fn)
        train_dataset = train_dataset.fillna(train_dataset.mean())  # 缺失值填充

        hour_data = train_dataset[train_dataset['TRI'] > 5].hour.values
        st_t = hour_data[hour_data < 12].min() #日出时间
        ed_t = hour_data[hour_data > 15].max() # 日落时间

        for col in train_dataset.columns:
            X = train_dataset[col].values.copy()

            X_min = st_min_max_dic[col][0]
            X_max = st_min_max_dic[col][1]
            X[X > 3000] = X_min
            X[X < X_min] = X_min
            X[X > X_max] = X_max
            X1 = (X - X_min)/(X_max-X_min + 1e-7)
            train_dataset[col] = X1

        ########################## 交叉验证 cv：1-7
        for cv_step in range(0,4):
            print(cv_step)

            if station_name == 'dazi':
                print('DAZI!!!!!!!!!!!!!!!!!!!!!!')
                y = train_dataset.iloc[:,0].values
                y_train = y[:-96].copy()
                y_test = y[-96:].copy()

                tun_xgb = np.array((dazi_train*18)[:370])/st_min_max_dic['TRI'][1]
                ans_xgb = np.array((dazi_test*2))/st_min_max_dic['TRI'][1]

                tun_lgb = np.array((dazi_train*18)[:370])/st_min_max_dic['TRI'][1]
                ans_lgb = np.array((dazi_test*2))/st_min_max_dic['TRI'][1]

            else:
                x_train, x_test, y_train, y_test = get_train_test(train_dataset, cv_step)

                print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

                params_lgb = {
                    'boosting_type': 'gbdt',
                    'objective': 'regression_l1',
                    'metric': 'mae',
                    'num_threads':4}
                model_lgb = lgb.LGBMRegressor(**params_lgb)
                params_test1 = {
                                'max_depth': [3,5,7,10],
                                'learning_rate': [0.001,0.1],
                                'min_data_in_leaf': [3,5,10],
                                'num_leaves': [10,20,40],
                                'min_child_weight':[5,10,20,30]
                                }
                gsearch1 = GridSearchCV(estimator=model_lgb, param_grid=params_test1,
                    scoring='neg_mean_squared_error', verbose=1, cv=4)
                gsearch1.fit(x_train, y_train)
                print(gsearch1.best_params_, gsearch1.best_score_)

                tunning_params = gsearch1.best_params_

                rg_lgb = lgb_model(x_train, y_train, x_test, y_test,tunning_params)

                # print(pd.DataFrame({
                #     'column': train_dataset.columns[1:],
                #     'importance': -1*rg_lgb.feature_importance(),
                # }).sort_values(by='importance'))

                rg_xgb = xgb_model(x_train, y_train, x_test, y_test)
                tun_xgb = predict_norm(rg_xgb, st_t, ed_t, x_train)
                ans_xgb = predict_norm(rg_xgb, st_t, ed_t, x_test)

                # if station_name == 'dazi':
                #     rg_lgb = lgb_model(x_train, y_train, x_test, y_test,1)
                # else:
                #     rg_lgb = lgb_model(x_train, y_train, x_test, y_test)

                tun_lgb = predict_norm(rg_lgb, st_t, ed_t, x_train)
                ans_lgb = predict_norm(rg_lgb, st_t, ed_t, x_test)

                # pickle.dump(rg_xgb, open('./models/' + versionNum + '/rg_xgb_'+ station_name + str(cv_step) +'.pickle', "wb"))
                pickle.dump(rg_lgb, open('./models/' + versionNum + '/rg_lgb_'+ station_name + str(cv_step) +'.pickle', "wb"))

            true_line = np.concatenate((y_train,y_test),0)*st_min_max_dic['TRI'][1]   #真实值变化
            xgb_line = np.concatenate((tun_xgb,ans_xgb),0)*st_min_max_dic['TRI'][1]   #xgb变化
            lgb_line = np.concatenate((tun_lgb,ans_lgb),0)*st_min_max_dic['TRI'][1]   #lgb变化

            ans = np.mean([ans_xgb, ans_lgb],0)

            mae_list = [mae_count(ans_xgb, y_test),mae_count(ans_lgb, y_test),mae_count(ans, y_test)]
            print(mae_list)

            station_line_plot([true_line,xgb_line,lgb_line], str(mae_list),
                out_pic = './op_pics/predict/'+ versionNum + '/' + station_name + str(cv_step) +  '_predict.jpg')

        mae_score.append(mae_list)

    mae_score = np.array(mae_score)
    score_all = 1/(1+np.mean(mae_score,0)*1100.0)
    print('Final Score is ::::::::::::::', score_all)

    station_mae_plot(mae_score, out_pic = './op_pics/predict/'+ versionNum + '/train_score_' + versionNum + '_test.jpg')
