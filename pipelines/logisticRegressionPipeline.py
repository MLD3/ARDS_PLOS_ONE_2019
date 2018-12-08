import time
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import LeaveOneOut, train_test_split, StratifiedKFold, GridSearchCV, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
import cProfile
import re
from io import  StringIO
import pstats
import argparse
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def getMedianModel(li):
    med = np.median([l[1] for l in li])
    shifted = [np.abs(l[1] - med) for l in li]
    medianIDX = np.argmin(shifted)
    return li[medianIDX][0]


def newPipe(jobObject, iters=5, regularization="l2"):
    experimentDict = {}
    # Parameter Grid for hyper-parameter tuning
    paramGrid = {'C': np.logspace(-4, 4, num=10)}
    splits = 5 # Number of folds in Repeated Stratified K-Fold CV (RSKFCV)
    repeats = 5 # Number of repeats in Repeated Stratified K-Fold CV (RSKFCV)
    experimentDict["paramGrid"] = paramGrid
    experimentDict["RSKFCV splits"] = splits
    experimentDict["RSKFCV repeats"] = repeats
    experimentDict["regularization"] = regularization
    for iteration in range(iters):
        print("iteration {} of {}".format(iteration, iters))
        dict_i = {}
        xTrainVal, yTrainVal = jobObject.getTrainValData()
        # store experiment information on first iteration
        if iteration == 0:
            experimentDict["xTrainVal"] = xTrainVal
            experimentDict["yTrainVal"] = yTrainVal
        rskf = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats)
        # Store model perf on train and val data for model with each hyper-parameter assignment for all train/val splits
        trainRows = []
        valRows = []
        for train_index, validation_index in rskf.split(xTrainVal, yTrainVal):
            # Separate train and val data for a single run in Repeated Stratified K-Fold CV (RSKFCV)
            xTrain = xTrainVal[train_index]
            yTrain = yTrainVal[train_index]
            xVal = xTrainVal[validation_index]
            yVal = yTrainVal[validation_index]
            # Store performance in train and val data for each hyper-parameter assignment
            trainRow = []
            valRow = []
            for cNum, c in enumerate(paramGrid["C"]):
                if regularization == "l2":
                    logReg = LogisticRegression(penalty="l2", class_weight='balanced', C=c)
                elif regularization == "l1":
                    logReg = LogisticRegression(penalty="l1", class_weight='balanced', C=c)
                else:
                    assert False, "{} regularization is not supported".format(regularization)
                logReg.fit(xTrain, yTrain)
                trainProbs = logReg.predict_proba(xTrain)
                trainAUC = roc_auc_score(yTrain, trainProbs[:,1], average="weighted")
                valProbs = logReg.predict_proba(xVal)
                valAUC = roc_auc_score(yVal, valProbs[:,1], average="weighted")
                # store the performance for this c val on this train val split
                trainRow.append(trainAUC)
                valRow.append(valAUC)
            # store the performance for this train/val split
            valRows.append(valRow)
            trainRows.append(trainRow)
        # From results of RSKFCV figure out optimal c-value
        trainRows = np.array(trainRows)
        valRows = np.array(valRows)
        trainMean = np.mean(trainRows, axis=0)
        valMean = np.mean(valRows, axis=0)
        chosenCIDX = np.argmax(valMean)
        chosenC = paramGrid["C"][chosenCIDX]
        dict_i["chosen c value"] = chosenC
        dict_i["cv train aucs"] = trainRows
        dict_i["cv val aucs"] = valRows
        # Retrain model using all train and validation data using optimal C value
        if regularization == "l2":
            fullLogReg = LogisticRegression(penalty="l2", class_weight='balanced', C=chosenC)
        elif regularization == "l1":
            fullLogReg = LogisticRegression(penalty="l1", class_weight='balanced', C=chosenC)
        else:
            assert False, "{} regularization is not supported".format(regularization)
        fullLogReg.fit(xTrainVal, yTrainVal)
        dict_i["full model coefficients"] = fullLogReg.coef_
        dict_i["full model intercept"] = fullLogReg.intercept_
        dict_i["full model n_iter_"] = fullLogReg.n_iter_
        # get bootstrapped test set for this iteration
        xTest, yTest = jobObject.getBootstrappedTest(iteration)
        dict_i["xTest"] = xTest
        dict_i["yTest"] = yTest
        # get predictions for test set
        testProbs = fullLogReg.predict_proba(xTest)
        dict_i["testProbs"] = testProbs
        # Calculate Test Performance
        testAUC = roc_auc_score(yTest, testProbs[:,1], average="weighted")
        dict_i["testAUC"] = testAUC
        jobObject.saveIteration(dict_i)
    return experimentDict
