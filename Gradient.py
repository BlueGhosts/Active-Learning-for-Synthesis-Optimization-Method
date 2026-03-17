# by wangjiaze on 2021-Apr-24
import numpy as np
import pandas as pd
from Sampling import CalculateDistanceMatrix


def PredictionByGradient(xtrain, ytrain, xtest):
    def CalculateGradientIncrement(gradientColumn, distance):
        gradientColumnMatrix = np.array((gradientColumn.T).tolist() * distance.shape[1]).reshape(-1, gradientColumn.shape[0]).T
        columnPrediction = gradientColumnMatrix*distance
        return columnPrediction

    def CalculateTestPrediction(xtrain, ytrain, xtest):
        gradients = CalculateGradient(xtrain, ytrain)
        distanceMatrix = pd.DataFrame(CalculateDistanceMatrix(xtest, xtrain), index=xtrain.index, columns=xtest.index)


        distanceMatrix[distanceMatrix == 0] = None
        distanceMatrix[distanceMatrix >= distanceMatrix.min() * 2] = None


        increment = pd.DataFrame(np.zeros([gradients.shape[0], distanceMatrix.shape[1]]), index=gradients.index, columns=distanceMatrix.columns)
        weight = 1 / distanceMatrix ** 2
        for column in xtest.columns:
            columnGradientIncrement = CalculateGradientIncrement(gradients.loc[:, column], distanceMatrix)
            increment += columnGradientIncrement ** 2
        increment = np.sqrt(increment)

        valueColumn = np.array((ytrain.T).tolist() * increment.shape[1]).reshape(increment.shape[1], -1).T
        allColumnPrediction = valueColumn + increment
        # allColumnPrediction = valueColumn
        # allColumnPrediction = increment
        weightedMeanPrediction = (allColumnPrediction * weight).sum() / weight.sum()
        return weightedMeanPrediction

    prediction = CalculateTestPrediction(xtrain, ytrain, xtest)
    prediction.loc[xtrain.index] = ytrain
    return prediction


def CalculateGradient(condition, value):
    def CalculateDifference(matrix1, matrix2):
        matrix1 = np.array((matrix1.T).tolist() * matrix2.shape[0]).reshape(-1, matrix1.shape[0]).T
        matrix2 = np.array((matrix2.tolist() * matrix1.shape[0])).reshape(matrix1.shape[0], -1)
        return matrix1 - matrix2


    distanceMatrix =  pd.DataFrame(CalculateDistanceMatrix(condition, condition), index=condition.index, columns=condition.index)
    valueDifferecnce = abs(pd.DataFrame(CalculateDifference(value, value), index=value.index, columns=value.index))
    gradients = pd.DataFrame(np.zeros_like(condition), index=condition.index, columns=condition.columns)


    distanceMatrix[distanceMatrix == 0] = None
    distanceMatrix[distanceMatrix >= distanceMatrix.min() * 1.5] = None

    weight = 1 / distanceMatrix
    for column in condition.columns:
        matrix = condition.loc[:, column]
        differenceColumn = pd.DataFrame(CalculateDifference(matrix, matrix), index=matrix.index, columns=matrix.index)
        allColumnGradient = valueDifferecnce * differenceColumn / (distanceMatrix**2)

        weightedMeanGradient = (allColumnGradient * (weight)).replace([np.inf, -np.inf], np.nan).sum() / (weight).replace([np.inf, -np.inf], np.nan).sum()

        gradients.loc[:, column] = weightedMeanGradient
    return gradients



if __name__ == '__main__':
    from Combination import Combination
    from ExperimentSpace import ExperimentSpace
    from Picture import Picture


    def ReadDonePointInformationCsv(filename):
        information = pd.read_csv(filename)
        condition = information.iloc[:, :-2]
        type = information.loc[:, 'class']
        value = information.loc[:, 'value']
        return condition, type, value


    # parameter_range = { 'P/Al': [0.0, 1.0, 0.05],
    #                      'template': [0.0, 1.0, 0.05],
    #                      # 'H20': [50, 100, 5],
    #                     }

    parameter_range = { 'P/Al': [0.0, 1.0, 0.01],
                         'template': [0.0, 1.0, 0.01],
                         # 'H20': [50, 100, 5],
                        }
    experimentCombination = pd.DataFrame(Combination(parameter_range.values(), 2), columns=parameter_range.keys())
    experimentSpace = ExperimentSpace(experimentCombination)

    donePointCondition, donePointType, donePointValue = ReadDonePointInformationCsv('TestGradient.csv')

    experimentSpace = ExperimentSpace(experimentCombination)
    experimentSpace.DonePoints(donePointCondition, donePointType, donePointValue)
    # print(experimentSpace.condition)
    # donePointCondition, donePointType, donePointValue = ReadDonePointInformationCsv('TestDoneGradient.csv')
    # print(experimentSpace.doneCondition())
    # print(experimentSpace.doneValue())


    targetType = 1
    targetTypeIndex = experimentSpace.type[experimentSpace.type == targetType].index
    preditcion = PredictionByGradient(experimentSpace.doneCondition().loc[targetTypeIndex, :],
                                      experimentSpace.doneValue().loc[targetTypeIndex],
                                      experimentSpace.condition)

    # print(preditcion)


    figure = Picture()
    ax1 = figure.addSubplot()
    ax1.addDataset(experimentSpace.doneCondition(), experimentSpace.doneValue())

    ax2 = figure.addSubplot()
    xy = experimentSpace.condition.loc[:, ]
    xy['preditcion'] = preditcion
    # print(xy)
    ax2.addDataset(xy, preditcion)

    figure.draw()
