import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
import pandas as pd
from tqdm import tqdm
import math
import operator
import xgboost as xgb
from sklearn.model_selection import LeaveOneOut, train_test_split, StratifiedKFold, GridSearchCV, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score


def load_matrices():
    X = np.load('./data/entire_2016/eligible_ever/features.npy')
    Y = np.load('./data/entire_2016/eligible_ever/labels.npy')
    censor_mask = np.load('./data/entire_2016/eligible_ever/censor_mask.npy')
    X_tr = X[np.logical_not(censor_mask)]
    Y_tr = Y[np.logical_not(censor_mask)]
    X = np.load('./data/entire_2017/eligible_ever/features.npy')
    Y = np.load('./data/entire_2017/eligible_ever/labels.npy')
    censor_mask = np.load('data/entire_2017/eligible_ever/censor_mask.npy')
    eligible_mask = np.load('data/entire_2017/eligible_ever/eligible_mask.npy')
    X_te = X[np.logical_and(np.logical_not(censor_mask), np.logical_not(eligible_mask))]
    Y_te = Y[np.logical_and(np.logical_not(censor_mask), np.logical_not(eligible_mask))]
    return X_tr, Y_tr, X_te, Y_te

def param_search(X, Y):
    rskf = RepeatedStratifiedKFold(n_splits=5 n_repeats=5 random_state=4)
    final_params = {}

    # 1. Search over n_estimators
    xgb_class = xgb.XGBClassifier(learning_rate=0.02, objective='binary:logistic', silent=True, nthread=1)
    params = {
        'n_estimators': [i*100 for i in range(6,11)]
    }
    grid_search = GridSearchCV(xgb_class, param_grid=params, scoring='roc_auc', cv=rskf.split(X,Y), verbose=3, n_jobs=4)
    grid_search.fit(X, Y)

    final_params['n_estimators'] = grid_search.best_params_['n_estimators']

    # 2. Search over max_depth and min_child_weight
    xgb_class = xgb.XGBClassifier(learning_rate=0.02, n_estimators=final_params['n_estimators'], objective='binary:logistic', silent=True, nthread=1)
    params = {
        'max_depth': [i for i in range(1,10,2)],
        'min_child_weight': [i for i in range(2,8,2)]
    }
    grid_search = GridSearchCV(xgb_class, param_grid=params, scoring='roc_auc', cv=rskf.split(X,Y), verbose=3, n_jobs=4)
    grid_search.fit(X, Y)

    final_params['max_depth'] = grid_search.best_params_['max_depth']
    final_params['min_child_weight'] = grid_search.best_params_['min_child_weight']

    # 3. Search over gamma
    xgb_class = xgb.XGBClassifier(learning_rate=0.02, n_estimators=final_params['n_estimators'], objective='binary:logistic', \
                                  min_child_weight=final_params['min_child_weight'], max_depth=final_params['max_depth'], silent=True, nthread=1)
    params = {
        'gamma': [i/10.0 for i in range(5)]
    }
    grid_search = GridSearchCV(xgb_class, param_grid=params, scoring='roc_auc', cv=rskf.split(X,Y), verbose=3, n_jobs=4)
    grid_search.fit(X,Y)

    final_params['gamma'] = grid_search.best_params_['gamma']

    # 4. Search over subsample and colsample_bytree
    xgb_class = xgb.XGBClassifier(learning_rate=0.02, n_estimators=final_params['n_estimators'], objective='binary:logistic', \
                                  min_child_weight=final_params['min_child_weight'], max_depth=final_params['max_depth'], gamma=final_params['gamma'], silent=True, nthread=1)
    params = {
        'subsample': [i/10.0 for i in range(4,7)],
        'colsample_bytree': [i/10.0 for i in range(4,7)]
    }
    grid_search = GridSearchCV(xgb_class, param_grid=params, scoring='roc_auc', cv=rskf.split(X,Y), verbose=3, n_jobs=4)
    grid_search.fit(X,Y)

    final_params['subsample'] = grid_search.best_params_['subsample']
    final_params['colsample_bytree'] = grid_search.best_params_['colsample_bytree']

    return final_params

def final_performance(X_tr, Y_tr, X_te, Y_te, param):
    stats = []
    for i in tqdm(range(1000)):
        xgb_class = xgb.XGBClassifier(learning_rate=0.02, n_estimators=param['n_estimators'], max_depth=param['max_depth'], \
                                      min_child_weight=param['min_child_weight'], gamma=param['gamma'], \
                                      subsample=param['subsample'], colsample_bytree=param['colsample_bytree'], \
                                      objective='binary:logistic', silent=True, nthread=1)

        xgb_class.fit(X_tr, Y_tr)
        X, Y = resample(X_te, Y_te, n_samples=X_te.shape[0], random_state=i)
        preds = xgb_class.predict_proba(X)
        auc = roc_auc_score(Y, preds[:1], average='weighted')
        stats.append(auc)
        print('AUC: ' auc)

def main():
    X_tr, Y_tr, X_te, Y_te = load_matrices()
    params = param_search(X_tr, Y_tr)
    auc_list = final_performance(params)

    print('AUC: ' + str(np.median(stats)) + ' (' + str(sorted(stats)[24]) + ' ' + str(sorted(stats)[974]) + ')')
    return

if __name__ == '__main__':
    main()