# python ExperimentSpace v1.0
# by wangjiaze on 2024-Feb-29

import pandas as pd
import numpy as np
from sklearn import preprocessing


class ExperimentSpace():
    def __init__(self, initValue, ps):
        # print(ps.zoom)
        # print(ps.sameThreshold)
        self.sameThreshold = ps.sameThreshold
        # self.zoom = ps.zoom


        self.originCondition = initValue

        self.zoomModel = preprocessing.MinMaxScaler().fit(self.originCondition)

        self.condition = ZoomDataFrame(self.originCondition, self.zoomModel)

        self.index = np.array(self.condition.index)
        self.number = self.index.shape[0]

        self.type = pd.Series([None] * self.number, index=self.index)
        self.value = pd.Series([None] * self.number, index=self.index)
        self.gradient = pd.Series([None] * self.number, index=self.index)

        self.distance = pd.Series([None] * self.number, index=self.index)
        self.stable = pd.Series([None] * self.number, index=self.index)
        self.value = pd.Series([None] * self.number, index=self.index)

        # self.type = pd.Series(np.zeros_like(self.index), index=self.index)
        # self.value = pd.Series(np.zeros_like(self.index), index=self.index)
        # self.gradient = pd.Series(np.zeros_like(self.index), index=self.index)

        self.typePrediction = self.type.copy()
        # self.typePrediction.to_string()

        # self.typePrediction = self.typePrediction.astype('string')
        self.valuePrediction = self.value.copy()
        self.gradientPrediction = self.gradient.copy()

        self.done = pd.Series(np.array([False] * self.number), index=self.index)

    def doneCondition(self):
        return self.condition.loc[self.donePointIndex(), :]

    def donePointIndex(self):
        return self.done[self.done == True].index

    def doneType(self):
        return self.type.loc[self.donePointIndex()]

    def doneValue(self):
        return self.value.loc[self.donePointIndex()]

    def undoneCondition(self):
        return self.condition.loc[self.undonePointIndex(), :]

    def undonePointIndex(self):
        return self.done[self.done == False].index

    def undoneType(self):
        return self.type.loc[self.undonePointIndex()]

    def undoneValue(self):
        return self.value.loc[self.undonePointIndex()]

    def typeNumber(self):
        return len(self.type.value_counts(normalize=False, dropna=True))

    def positivePointNumber(self, threshold = 0):
        return self.value[self.value > threshold].shape[0]



    # def GetPointConditions(self, index):
    #     return self.condition.loc[index, :]

    def seletePoint(self, index):
        self.done.loc[index] = True

    def doPoints(self, conditions, types, values):
        for pointIndex in conditions.index:
            condition = conditions.loc[pointIndex, :]
            type = types.loc[pointIndex]
            value = values.loc[pointIndex]
            self.doPoint(condition, type, value)

    def doPoint(self, condition, type, value):
        index = self.checkAndAddCondition(condition)
        self.type.loc[index] = type
        self.value.loc[index] = value
        self.done.loc[index] = True

    def checkAndAddCondition(self, condition):
        copyCondition = self.originCondition.loc[:]
        for i in range(copyCondition.shape[1]):
            # copyCondition = copyCondition[abs(copyCondition.iloc[:, i] - condition[i])<=self.sameThreshold]
            # print(condition)
            copyCondition = copyCondition[abs(copyCondition.iloc[:, i] - condition[i]) <= self.sameThreshold]

        if copyCondition.shape[0] == 0:
            index = self.addPoint(condition)
        else:
            index = copyCondition.index
        return index

    def addPoint(self, condition):
        # newIndex = self.index.tolist()[-1] + 1
        newIndex = self.index.tolist()[-1] + "_1" 
        self.originCondition.loc[newIndex, :] = condition
        self.condition.loc[newIndex, :] = self.zoomModel.transform(np.array(condition).reshape(1, -1))

        self.index = np.array(self.condition.index)
        self.number = self.condition.shape[0]

        self.type.loc[newIndex] = 0
        self.value.loc[newIndex] = 0
        self.gradient.loc[newIndex] = 0

        self.typePrediction.loc[newIndex] = 0
        self.valuePrediction.loc[newIndex] = 0
        self.gradientPrediction.loc[newIndex] = 0

        self.done.loc[newIndex] = False
        return newIndex


class PartExperimentSpace(ExperimentSpace):
    def __init__(self, father, index):
        self.father = father

        self.originCondition = father.originCondition.loc[index, :]
        self.zoomModel = father.zoomModel

        self.condition = father.condition.loc[index, :]

        self.index = np.array(self.condition.index)
        self.number = self.index.shape[0]

        self.type = father.type.loc[index]
        self.value = father.value.loc[index]
        self.gradient = father.gradient.loc[index]

        self.typePrediction = father.typePrediction.loc[index]
        self.valuePrediction = father.valuePrediction.loc[index]
        self.gradientPrediction = father.gradientPrediction.loc[index]
        self.done = father.done.loc[index]


def ZoomDataFrame(dataset, model = ''):
    if not model:
        model = preprocessing.MinMaxScaler().fit(dataset)
    afterDataset = pd.DataFrame(model.transform(dataset), index=dataset.index, columns=dataset.columns)
    return afterDataset


def InverseZoomDataFrame(dataset, model):
    return model.inverse_transform(dataset)


def CalculateDistanceMatrix(points):
    def compute_squared_EDM_method3(X):
        # determin dimensions of data matrix
        m, n = X.shape
        # compute Gram matrix
        G = np.dot(X.T, X)
        # initialize squared EDM D
        D = np.zeros([n, n])
        # iterate over upper triangle of D
        for i in range(n):
            for j in range(i + 1, n):
                d = X[:, i] - X[:, j]
                D[i, j] = G[i, i] - 2 * G[i, j] + G[j, j]
                D[j, i] = D[i, j]
        return D

    def compute_squared_EDM_method4(X):
        # https://blog.csdn.net/justin18chan/article/details/79350354
        m, n = X.shape
        G = np.dot(X.T, X)
        H = np.tile(np.diag(G), (n, 1))
        return H + H.T - 2 * G

    # d = compute_squared_EDM_method3(points)
    distanceMatrix = compute_squared_EDM_method4(points.T)
    # print(distanceMatrix)
    return distanceMatrix




if __name__ == '__main__':
    # Test  ##
    parameter_range = { 'P/Al': [0.5, 1.0, 0.1],
                         'template': [1, 10, 1],
                         'H20': [50, 100, 50],
                        }

    # parameter_range = { 'P/Al': [0.5, 1.0, 0.01],
    #                      'template': [1, 10, 0.1],
    #                      'H20': [50, 100, 5],
    #                     }
    from Combination import Combination

    experimentCombination = pd.DataFrame(Combination(parameter_range.values(), 2), columns=parameter_range.keys())
    experimentSpace = ExperimentSpace(experimentCombination)
    # print(experimentSpace.condition)
    print(experimentSpace.undoneCondition())
    print(experimentSpace.doneCondition())
    print(experimentSpace.undoneCondition())

    # experimentSpace.distanceMatrix()

