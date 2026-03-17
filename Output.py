
import pandas as pd

def Output(outFilepath, nextPoints, experimentSpace):
    def MainOutput():
        pass

    nextPoints.to_csv(outFilepath)

    # print(outFilepath)
    # print(nextPoints)

    conditionPrediction = experimentSpace.originCondition
    conditionPrediction['typePrediction'] = experimentSpace.typePrediction

    # conditionPrediction = experimentSpace.originCondition
    conditionPrediction['valuePrediction'] = experimentSpace.valuePrediction
    # conditionPrediction['valuePrediction']

    conditionPrediction.fillna(0, inplace=True)
    conditionPrediction['gradientPrediction'] = experimentSpace.gradientPrediction

    # conditionPrediction['distance'] = experimentSpace.distance
    # conditionPrediction.fillna('/', inplace=True)
    # conditionPrediction.fillna('/', inplace=True)
    # conditionPrediction['Prediction'] = experimentSpace.valuePrediction


    # print(conditionPrediction)
    # print(conditionPrediction[conditionPrediction.loc[:, 'valuePrediction'] > 0])
    #
    #
    # print(conditionPrediction.loc[nextPoints.index, :])
    #
    # print(conditionPrediction.distance)
    # print(conditionPrediction[conditionPrediction.loc[:, 'distance'] > 0])

    # psCorrectItems = abs(experimentSpace.gradientPrediction - experimentSpace.valuePrediction)  # prediction Stable
    # pvCorrectItems = experimentSpace.gradientPrediction + experimentSpace.valuePrediction  # prediction Value


    # print(conditionPrediction[conditionPrediction.loc[:, 'gradientPrediction'] != '/'])

    experimentSpace.originCondition.to_csv(outFilepath, sep=',')
    return



if __name__ == '__main__':
    pass