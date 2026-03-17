# by wangjiaze on 2021-Apr-24
# V1.1 by wangjiaze in 2022-Nov-24  replace correctItems by psCorrectItems, pvCorrectItems

import pandas as pd
import numpy as np
from sklearn import preprocessing

np.set_printoptions(precision=16, threshold=None, edgeitems=None, linewidth=None, suppress=None, nanstr=None, infstr=None, formatter=None)


# def SampleOnePoint(donePoints, undonePoints, correctItems = pd.DataFrame(), startPointWay = 'random', zoom = False):
def SampleOnePoint(donePoints, undonePoints, psCorrectItems=pd.DataFrame(), pvCorrectItems=pd.DataFrame(), sampleFactor_weight = {'distance':0.5, 'predictionStable':0.5, 'predictionValue':0}, startPointWay='random', zoom=False):
    def GetStartIndex(points, startPointWay):
        def MedianPoint(points):
            point = points.loc[:]
            if points.shape[0] % 2 == 1:
                median = np.median(points, axis=0)
            else:
                median = np.median(points.append(points.iloc[0]), axis=0)
            for i in range(points.shape[1]):
                point = point[abs(point.iloc[:, i] -  median[i]) <= 0.0001]
            return point

        if startPointWay == 'median':
            startIndex = MedianPoint(points).index
        elif startPointWay == 'random':
            startIndex = points.sample(1).index
        else:
            startIndex = None
            print('Sampling StartPoint: \'median\',\'random\' be asked!')
        return startIndex.tolist()[0]
    if len(donePoints) == 0:
        index = GetStartIndex(undonePoints, startPointWay)
        # print(index)
    else:
        if zoom:
            zoomModel = preprocessing.MinMaxScaler().fit(undonePoints)
            donePoints = pd.DataFrame(zoomModel.transform(donePoints), index=donePoints.index, columns=donePoints.columns)
            undonePoints = pd.DataFrame(zoomModel.transform(undonePoints), index=undonePoints.index, columns=undonePoints.columns)
        # index = FarthestPointSampling(donePoints, undonePoints, correctItems)
        index = FarthestPointSampling(donePoints, undonePoints, psCorrectItems, pvCorrectItems, sampleFactor_weight)
    return index


# def FarthestPointSampling(donePoints, undonePoints, correctItems = pd.DataFrame()):
def FarthestPointSampling(donePoints, undonePoints, psCorrectItems=pd.DataFrame(), pvCorrectItems=pd.DataFrame(), sampleFactor_weight = {'distance':0.5, 'predictionStable':0.5, 'predictionValue':0}):
    def CorrectDistance(neatestDistance, psCorrectItems, pvCorrectItems):
        zoomNeatestDistance = (neatestDistance - neatestDistance.min()) / (neatestDistance.max() - neatestDistance.min())

        psCorrectItems = psCorrectItems.replace(np.nan, 0)
        pvCorrectItems = pvCorrectItems.replace(np.nan, 0)

        correctionDistance = sampleFactor_weight['distance']*zoomNeatestDistance + sampleFactor_weight['predictionStable']*psCorrectItems + sampleFactor_weight['predictionValue']*pvCorrectItems
        return correctionDistance

    distanceMatrix = CalculateDistanceMatrix(donePoints, undonePoints)
    distanceMatrix = pd.DataFrame(distanceMatrix, index=undonePoints.index)

    neatestDistance = distanceMatrix.min(axis=1)
    if psCorrectItems.shape[0] != 0:
        # print(psCorrectItems)
        psCorrectItems = (psCorrectItems - psCorrectItems.min()) / (psCorrectItems.max() - psCorrectItems.min())
        pvCorrectItems = (pvCorrectItems - pvCorrectItems.min()) / (pvCorrectItems.max() - pvCorrectItems.min())

        correctionDistance = CorrectDistance(neatestDistance, psCorrectItems, pvCorrectItems)
        # correctionDistance = correctionDistance.dropna(how='all')
    else:
        correctionDistance = neatestDistance


    index = correctionDistance.replace(np.nan, 0).idxmax()


    print(index)
    print(neatestDistance[index])
    undonePoints.loc[index,:].distance = neatestDistance[index]

    # index = correctionDistance.replace(np.nan, 0).idxmax(axis = 1)
    return index


def CalculateDistanceMatrix(matrix1, matrix2):
    ##  程序待检测  ##
    # https://blog.csdn.net/frankzd/article/details/80251042
    distanceMatrix = np.sqrt(abs(-2 * np.dot(matrix2, matrix1.T) + [np.sum(np.square(matrix1), axis=1).tolist()] * matrix2.shape[0]
                             + np.transpose([np.sum(np.square(matrix2), axis=1)])))
    return distanceMatrix


if __name__ == '__main__':
    pass