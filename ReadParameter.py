import configparser

# 读取配置文件
# config = configparser.RawConfigParser()
# config.read("parameter.ini")
#
# # 获取文件的所有section
# secs = config.sections()
# print(secs)
#
# # 获取指定section下的所有参数key
# options = config.options("Project")
# print(options)
#
# # 获取指定section中指定key的value
# param = config.get("Classification", "model")
# print(param)


class Parameter():
    def __init__(self, environmentFile):
        config = configparser.RawConfigParser()
        config.read(environmentFile)

        self.project = Project(config['Project'])
        self.calculate = Calculate(config['Calculate'])
        self.space = Space(config['ConditionSpace'])
        self.classification = Classification(config['Classification'])
        self.regression = Regression(config['Regression'])


class Project(Parameter):
    def __init__(self, configure):
        self.projectName = configure.get("projectName")
        self.conditionSpaceFilepath = configure.get("conditionSpace")
        self.knownConditionFilepath = configure.get("knownCondition")
        self.outFilepath = configure.get("outFilepath")
        self.outCondition = configure.get("outCondition")

        self.regressionStartThreshold = configure.getint("regressionStartThreshold")
        self.regressionSampleRadio = configure.getfloat("regressionSampleRadio")

        self.gradientStartThreshold = configure.getint("gradientStartThreshold")



class Calculate(Parameter):
    def __init__(self, configure):
        self.target = configure.getint("target")
        self.planPointNumber = configure.getint("planPointNumber")

        self.regressionStartThreshold = configure.getint("regressionStartThreshold")
        self.regressionSampleRadio = configure.getfloat("regressionSampleRadio")
        self.gradientStartThreshold = configure.getint("gradientStartThreshold")

        self.distanceWeight = configure.getfloat("distanceWeight")
        self.stableWeight = configure.getfloat("stableWeight")
        self.valueWeight = configure.getfloat("valueWeight")

        self.sameThreshold = configure.getfloat("sameThreshold")
        self.sampleZoom = configure.getboolean("sampleZoom")
        self.sampleStartWay = configure.get("sampleStartWay")


class Space(Parameter):
    def __init__(self, configure):
        # self.zoom = configure.getboolean("zoom")
        self.sameThreshold = configure.getfloat("sameThreshold")

class Classification(Parameter):
    def __init__(self, configure):
        # self.zoom = configure.getboolean("zoom")
        self.model = configure.get("model")
        self.kernel = configure.get("kernel")
        self.C = configure.getfloat("C")
        self.class_weight = configure.get("class_weight")


class Regression(Parameter):
    def __init__(self, configure):
        # self.zoom = configure.getboolean("zoom")
        self.model = configure.get("model")
        self.kernel = configure.get("kernel")
        self.C = configure.getfloat("C")
        self.class_weight = configure.get("class_weight")


if __name__ == '__main__':
    environmentFile = 'parameter-AXY.ini'

    parameter = Parameter(environmentFile)
    print(parameter.calculate.typeTarget)
    print(parameter.project.projectName)