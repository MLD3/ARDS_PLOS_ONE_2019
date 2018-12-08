import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelines import logisticRegressionPipeline
from jobObjects import Data, Pipeline, Job
import time
import numpy as np
from sklearn.model_selection import LeaveOneOut, train_test_split, StratifiedKFold, GridSearchCV, RepeatedStratifiedKFold
import multiprocessing as mp
from sklearn.utils import resample

def runJob(job):
    job.run()

def getTrainVal(self):
    xTrainVal = self.trainData.matrices["features"]
    yTrainVal = self.trainData.matrices["labels"]
    earlyARDSMask = self.trainData.matrices["censor_mask"]
    eligibleMaskTrainVal = self.trainData.matrices["eligible_mask"]
    # Remove those patients from training set that develop ARDS prior to six hours
    x = xTrainVal[np.logical_not(earlyARDSMask)]
    y = yTrainVal[np.logical_not(earlyARDSMask)]
    return x, y

def getTestSet(self, splitNumber):
    xTest = self.testData.matrices["features"]
    yTest = self.testData.matrices["labels"]
    # mask to remove patients who developed ARDS prior to six hours
    earlyARDSMask = self.testData.matrices["censor_mask"]
    # mask To remove patients who became eligible prior to six hours
    eligibleMask = self.testData.matrices["eligible_mask"]
    mask = np.logical_and(np.logical_not(earlyARDSMask), np.logical_not(eligibleMask))
    xPossible, yPossible = xTest[mask], yTest[mask]
    x, y = resample(xPossible,yPossible, replace=True, random_state=splitNumber)
    return x, y

trainData = Data("../../NEW_matrices/entire_2016/eligible_ever/",
        ["features.npy", "labels.npy", ("eligible_mask.npy", "bool"), "dict.npy", ("censor_mask.npy", "bool")],
        "Patients from Jan - March 2016 that don't develop ARDS before 6 hours and meet risk stratification eligibility criteria within first 7 hospital days")

testData = Data("../../NEW_matrices/entire_2017/eligible_ever/",
                ["features.npy", "labels.npy", ("eligible_mask.npy", "bool"), "dict.npy", ("censor_mask.npy", "bool")],
                "Patients from Jan - March 2017 that don't develop ARDS before 6 hours and meet risk stratification eligibility criteria after being observed for at least 6 hours")


p = Pipeline(logisticRegressionPipeline.newPipe, "Most recent version of logistic regression pipeline")

savePath = "/data/dzeiberg/ards/experiments/"
nIterations = 1000
jobs = []
jobs.append(Job(trainData, testData, getTrainVal, getTestSet, nIterations, "l2", p, savePath, "running pipeline with L2 regularization"))
jobs.append(Job(trainData, testData, getTrainVal, getTestSet, nIterations, "l1", p, savePath, "running pipeline with L1 regularization"))
# run all jobs
pool = mp.Pool(processes=4)
results = [pool.apply_async(runJob, args=(job,)) for job in jobs]
for p in results:
    p.get()
