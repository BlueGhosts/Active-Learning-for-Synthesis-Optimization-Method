
def Classification(xtrain, ytrain, xtest, para):
    if para.model == 'svm':
        from sklearn import svm
        classifyModel = svm.SVC(kernel=para.kernel, C=para.C, class_weight=para.class_weight)
    if para.model == 'xgboost':
        from xgboost import XGBClassifier as XGBC
        from xgboost import XGBRegressor as XGBR
        classifyModel = XGBC(class_weight=para.class_weight)

    prediction = classifyModel.fit(xtrain, ytrain).predict(xtest)
    return prediction


def Regression(xtrain, ytrain, xtest, para):
    if para.model == 'xgboost':
        from xgboost import XGBClassifier as XGBC
        from xgboost import XGBRegressor as XGBR
        regressModel = XGBR()
    elif para.model == 'svm':
        from sklearn import svm
        regressModel = svm.SVR(kernel="rbf", C=10.0)

    prediction = regressModel.fit(xtrain, ytrain).predict(xtest)
    return prediction


# def Classification0(xtrain, ytrain, xtest, model = 'svm'):
#     if model == 'svm':
#         classifyModel = svm.SVC(kernel="rbf", C=10.0, class_weight='balanced')
#     if model == 'xgboost':
#         classifyModel = XGBC(class_weight='balanced')
#
#     prediction = classifyModel.fit(xtrain, ytrain).predict(xtest)
#     return prediction
#
#
# def Regression0(xtrain, ytrain, xtest, model = 'xgboost'):
#     if model == 'xgboost':
#         regressModel = XGBR()
#     elif model == 'svm':
#         regressModel = svm.SVR(kernel="rbf", C=10.0)
#
#     prediction = regressModel.fit(xtrain, ytrain).predict(xtest)
#     return prediction

if __name__ == '__main__':
    pass