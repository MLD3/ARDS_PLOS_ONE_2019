import os
import numpy as np
import time
import pathlib

class Data(object):
    """
    Object representing a dataset
    path: absolute path to directory where files in fileNames list can be found
    fileNames: file names (with extension) for all matrices for this data set
    description: description of what this data is
    """
    def __init__(self, path, fileNames, desc):
        super(Data, self).__init__()
        self.path = path
        self.fileNames = fileNames
        self.desc = desc
        self.matrices = {}
        self.load()

    def load(self):
        for file in self.fileNames:
            # load this file
            if isinstance(file, tuple):
                filename = file[0]
                filePath = os.path.join(self.path, filename)
                if file[1] == "bool":
                    self.matrices[filename[:filename.find(".")]] = np.load(filePath).astype(bool)
                elif file[1] == "dict":
                    self.matrices[filename[:filename.find(".")]] = np.load(filePath).item()
                else:
                    assert False, "bad filename input"
            else:
                filePath = os.path.join(self.path, file)
                self.matrices[file[:file.find(".")]] = np.load(filePath)
    def getDescription(self):
        desc = self.desc + "\n" + "path: {}\n".format(self.path)
        for fileName in self.fileNames:
            desc += "\t{}".format(fileName)
        return desc
        


class Pipeline(object):
    """
    func: function object for this pipeline
    desc: description of what pipeline is being used, what it does, and what it returns.
    """
    def __init__(self, func, desc):
        super(Pipeline, self).__init__()
        self.func = func
        self.desc = desc
        

class Job(object):
    """
    Object representing a single run of some data object on some pipeline
    """
    def __init__(self, trainData, testData, getTrainVal, getTestSet, iters, regularization, pipeline, savePath, description):
        super(Job, self).__init__()
        self.trainData = trainData
        self.testData = testData
        self.getTrainVal = getTrainVal
        self.getTestSet = getTestSet
        self.iters = iters
        self.regularization = regularization
        self.pipeline = pipeline
        self.savePath = savePath
        self.runTime = None
        self.results = None
        self.description = description
        self.modSaveName()

    def getTrainValData(self):
        return self.getTrainVal(self)

    def getBootstrappedTest(self, splitNumber):
        return self.getTestSet(self, splitNumber)

    def run(self):
        startTime = time.time()
        self.results = self.pipeline.func(self, iters=self.iters, regularization=self.regularization)
        stopTime = time.time()
        self.runTime = stopTime - startTime
        self.save()

    def modSaveName(self):
        i = 1
        while pathlib.Path(self.savePath+"experiment_{}".format(i)).is_dir():
            i += 1
        pathlib.Path(self.savePath+"experiment_{}".format(i)).mkdir(parents=True)
        self.savePath = self.savePath+"experiment_{}/".format(i)

    def saveIteration(self, exp):
        iteration=1
        path = self.savePath+"iteration_{}.npy".format(iteration)
        while os.path.exists(path):
            iteration += 1
            path = self.savePath+"iteration_{}.npy".format(iteration)
        print("saving to {}".format(str(path)))
        np.save(str(path), exp)

    def save(self):
        """
        write a README file that contains:
            data description
            pipeline description
            job runtime
        """
        
        with open(os.path.join(self.savePath, "README"), "w") as f:
            f.write("Job Description\n{}\n\n".format(self.description))
            f.write("Pipeline Description\n"+self.pipeline.desc+"\n\n")
            f.write("Train Data Description\n"+self.trainData.getDescription()+"\n\n")
            f.write("Test Data Description\n"+self.testData.getDescription()+"\n\n")
            f.write("Job Runtime: {} seconds".format(self.runTime))
        np.save(os.path.join(self.savePath, "results"), self.results)
