# python ExperimentPlanMain v1.0
# by wangjiaze on 2024-Feb-28

from ReadParameter import Parameter
# def ReadExperimentCombinationCsv(filename):
#     return pd.read_csv(filename)

import pandas as pd
from ExperimentSpace import ExperimentSpace
from ExperimentPlan import ExperimentPlan
from Output import Output

def Plan(experimentSpace, parameter):
    # nextPointIndex = ExperimentPlan(experimentSpace,
    #                                 sampleNumber=pc.planPointNumber,
    #                                 regressionSampleRadio=pc.regressionSampleRadio,
    #                                 sampleFactor_weight=pc.sampleFactor_weight)

    nextPointIndex = ExperimentPlan(experimentSpace, parameter)
    return nextPointIndex

def main(parameter):
    def ReadExperimentCombinationCsv(filename):
        data = pd.read_csv(filename, index_col=0)
        return data

    def ReadDonePointInformationCsv(filename):
        information = pd.read_csv(filename, index_col=0)
        condition = information.iloc[:, :-2]
        type = information.loc[:, 'class']
        value = information.loc[:, 'value']
        return condition, type, value


    paraProject = parameter.project

    experimentSpace = ExperimentSpace(ReadExperimentCombinationCsv(paraProject.conditionSpaceFilepath), parameter.space)
    donePointCondition, donePointType, donePointValue = ReadDonePointInformationCsv(paraProject.knownConditionFilepath)
    experimentSpace.doPoints(donePointCondition, donePointType, donePointValue)
    nextPointIndex = Plan(experimentSpace, parameter)
    nextPoints = experimentSpace.originCondition.loc[nextPointIndex, :]
    print(nextPoints)
    nextPoints.to_csv(paraProject.outCondition)
    
    Output(paraProject.outFilepath, nextPoints, experimentSpace)
    return nextPoints



if __name__ == '__main__':
    # parameter = Parameter('SAPO-file/parameter-SAPO.ini')
    # parameter = Parameter('parameter-AXY.ini')
    parameter = Parameter('ZEO2-file/parameter-ZEO2.ini')
    print(parameter.project.projectName)

    main(parameter)
    # nextPoints = main(parameter)
    # print(nextPoints)
