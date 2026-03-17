# V1.1 by wangjiaze in 221109  replace correctItems by psCorrectItems, pvCorrectItems
# V1.1 by wangjiaze in 240307

from MLPredicition import Classification, Regression
from Sampling import SampleOnePoint

from ExperimentSpace import PartExperimentSpace
from ExperimentSpace import ExperimentSpace
from Gradient import PredictionByGradient
import pandas as pd
import numpy as np


# def ExperimentPlan(experimentSpace, samplingNumber = 1, target = 1, regressionSampleRadio = 1, sampleFactor_weight = {'distance':0.5, 'predictionStable':0.5, 'predictionValue':0}):


def ExperimentPlan(experimentSpace, parameter):
    paraCalculate = parameter.calculate
    paraClassification = parameter.classification
    paraRegression = parameter.regression

    classificationSpace = CalculateClassificationSpace(experimentSpace, paraClassification, paraCalculate.target)
    regressionSpace = CalculateValuePredictionSpace(experimentSpace, classificationSpace, paraCalculate, paraRegression)
    psCorrectItems, pvCorrectItems = CalculateCorrectItems(experimentSpace, classificationSpace, paraCalculate)
    sampleFactor_weight = {'distance': paraCalculate.distanceWeight,
                           'predictionStable': paraCalculate.stableWeight,
                           'predictionValue': paraCalculate.valueWeight}
    nextPointIndex = CalculateNextPointIndex(regressionSpace, experimentSpace, paraCalculate.planPointNumber, psCorrectItems,
                                             pvCorrectItems, sampleFactor_weight)
    return nextPointIndex


def ExperimentPlan0(experimentSpace, samplingNumber=1, target=1, regressionSampleRadio=1,
                    sampleFactor_weight={'distance': 0.5, 'predictionStable': 0.5, 'predictionValue': 0}):

    classificationSpace = CalculateClassificationSpace(experimentSpace, target)
    # print('## classificationSpace ##')
    # print(classificationSpace.condition)
    regressionSpace = CalculateValuePredictionSpace(experimentSpace, classificationSpace, regressionSampleRadio, target)
    # correctItems = CalculateCorrectItems(experimentSpace, classificationSpace)
    # nextPointIndex = CalculateNextPointIndex(regressionSpace, experimentSpace, samplingNumber, correctItems)
    psCorrectItems, pvCorrectItems = CalculateCorrectItems(experimentSpace, classificationSpace)
    nextPointIndex = CalculateNextPointIndex(regressionSpace, experimentSpace, samplingNumber, psCorrectItems, pvCorrectItems, sampleFactor_weight)
    return nextPointIndex


def CalculateClassificationSpace(experimentSpace, paraClassification, target = 1):
    if experimentSpace.typeNumber() > 1:
        experimentSpace.typePrediction.loc[experimentSpace.index] = Classification(experimentSpace.doneCondition(),
                                                                                   experimentSpace.doneType().astype('int'),
                                                                                   experimentSpace.condition, paraClassification)

        classificationIndex = experimentSpace.typePrediction[experimentSpace.typePrediction == target].index
        classificationSpace = PartExperimentSpace(experimentSpace, classificationIndex)
    else:
        classificationSpace = experimentSpace
    experimentSpace.originCondition.to_csv("ZEO2-classCondition-V2R3.csv",sep=',')
    experimentSpace.typePrediction.to_csv("ZEO2-classPrediciton-V2R3.csv",sep=',')
    return classificationSpace


def CalculateValuePredictionSpace(experimentSpace, classificationSpace, paraCalculate, paraRegression):
    def CalculateRegressionSpaceIndex(experimentSpace, classificationSpace, paraCalculate):
        if classificationSpace.positivePointNumber() >= paraCalculate.regressionStartThreshold:
            targetTypeIndex = classificationSpace.doneType()[classificationSpace.doneType() == paraCalculate.target].index
            regressionPrediciton = Regression(classificationSpace.doneCondition().loc[targetTypeIndex, :],
                                              classificationSpace.doneValue().loc[targetTypeIndex],
                                              classificationSpace.condition, paraRegression)
            classificationSpace.valuePrediction.loc[classificationSpace.index] = regressionPrediciton
            experimentSpace.valuePrediction.loc[classificationSpace.index] = regressionPrediciton

            regressionSpaceIndex = GetTopData(classificationSpace.valuePrediction, paraCalculate.regressionSampleRadio)
        else:
            regressionSpaceIndex = classificationSpace.index
        return regressionSpaceIndex

    def CalculateGradientSpaceIndex(experimentSpace, classificationSpace, paraCalculate):
        if classificationSpace.positivePointNumber() >= paraCalculate.gradientStartThreshold:
            targetTypeIndex = classificationSpace.doneType()[classificationSpace.doneType() == paraCalculate.target].index
            gradientPrediction = PredictionByGradient(classificationSpace.doneCondition().loc[targetTypeIndex, :],
                                                      classificationSpace.doneValue().loc[targetTypeIndex],
                                                      classificationSpace.condition)
            experimentSpace.gradientPrediction.loc[gradientPrediction.index] = gradientPrediction
            classificationSpace.gradientPrediction.loc[gradientPrediction.index] = gradientPrediction

            gradientSpaceIndex = GetTopData(classificationSpace.gradientPrediction, paraCalculate.regressionSampleRadio)
        else:
            gradientSpaceIndex = []
        return gradientSpaceIndex

    def GetTopData(data, radio):
        threshold = data[data > 0].quantile(q=1 - radio, interpolation='linear')
        topIndex = data[data >= threshold].index
        return topIndex

    regressionSpaceIndex = CalculateRegressionSpaceIndex(experimentSpace, classificationSpace, paraCalculate)

    gradientSpaceIndex = CalculateGradientSpaceIndex(experimentSpace, classificationSpace, paraCalculate)
    unionIndex = np.union1d(regressionSpaceIndex, gradientSpaceIndex)
    regressionSpace = PartExperimentSpace(experimentSpace, unionIndex)
    regressionSpace.gradientPrediction = experimentSpace.gradientPrediction.loc[regressionSpace.index]


    experimentSpace.originCondition.to_csv("ZEO2-valueCondition-V2R3.csv",sep=',')
    experimentSpace.valuePrediction.to_csv("ZEO2-valuePrediciton-V2R3.csv",sep=',')
    # experimentSpace.originCondition.to_csv("TEA-valueCondition-round2.csv",sep=',')
    # experimentSpace.valuePrediction.to_csv("TEA-valuePrediciton-round2.csv",sep=',')
    return regressionSpace


def CalculateCorrectItems(experimentSpace, classificationSpace, paraCalculate):
    if classificationSpace.positivePointNumber() >= min(paraCalculate.regressionStartThreshold, paraCalculate.gradientStartThreshold):
        experimentSpace.gradientPrediction = experimentSpace.gradientPrediction.replace("", 0)
        experimentSpace.gradientPrediction = experimentSpace.gradientPrediction.replace(np.nan, 0)

        psCorrectItems = abs(experimentSpace.gradientPrediction - experimentSpace.valuePrediction) # prediction Stable
        pvCorrectItems = experimentSpace.gradientPrediction + experimentSpace.valuePrediction      # prediction Value
    else:
        # correctItems = pd.DataFrame()
        psCorrectItems = pd.DataFrame()      # prediction Stable
        pvCorrectItems = pd.DataFrame()      # prediction Value
    return psCorrectItems, pvCorrectItems


# def CalculateNextPointIndex(sampleSpace, doneSpace, samplingNumber = 1, correctItems = pd.DataFrame()):
def CalculateNextPointIndex(sampleSpace, doneSpace, samplingNumber = 1, psCorrectItems = pd.DataFrame(), pvCorrectItems = pd.DataFrame(), sampleFactor_weight = {'distance':0.5, 'predictionStable':0.5, 'predictionValue':0}):

    def CalculatePredictedBestIndex(space):
        predictionDescendingIndex = space.valuePrediction.sort_values(ascending=False).index
        for undoBestIndex in predictionDescendingIndex:
            if undoBestIndex not in space.donePointIndex():
                space.seletePoint(undoBestIndex)
                return undoBestIndex
        return 'ERROR : every point in the space had been done!'

        
    # def Sampling(sampleSpace, doneSpace, samplingNumber=1, correctItems=pd.DataFrame(), zoom=True, startPointWay='median'):
    def Sampling(sampleSpace, doneSpace, samplingNumber = 1, psCorrectItems=pd.DataFrame(), pvCorrectItems=pd.DataFrame(), sampleFactor_weight = {'distance':0.5, 'predictionStable':0.5, 'predictionValue':0}, zoom=True, startPointWay='median'):
        samplingIndexes = []
        for i in range(samplingNumber):
            index = SampleOnePoint(doneSpace.doneCondition(), sampleSpace.undoneCondition(), psCorrectItems, pvCorrectItems, sampleFactor_weight, startPointWay, zoom)
            sampleSpace.seletePoint(index)
            doneSpace.seletePoint(index)
            samplingIndexes.append(index)
        return samplingIndexes

    def RandomSampling(sampleSpace, samplingNumber=1):
        samplingIndexes = np.random.choice(sampleSpace.undoneCondition().index, samplingNumber)
        samplingIndexes = [ int(index) for index in samplingIndexes]
        for index in samplingIndexes:
            sampleSpace.seletePoint(index)
            doneSpace.seletePoint(index)
        return samplingIndexes


    nextPointIndex = []
    if sampleSpace.valuePrediction.sum() != 0:
        nextPointIndex.append(CalculatePredictedBestIndex(sampleSpace))
        nextPointIndex += Sampling(sampleSpace, doneSpace, samplingNumber - 1, psCorrectItems, pvCorrectItems, sampleFactor_weight)
    else:
        nextPointIndex += Sampling(sampleSpace, doneSpace, samplingNumber, psCorrectItems, pvCorrectItems, sampleFactor_weight)

    return nextPointIndex


if __name__ == '__main__':
    pass