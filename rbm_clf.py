
from __future__ import print_function
import collections
import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import argparse
import sqlite3
import datetime
import time
import json
import copy
from import_db import *
from import_txt import *

data_path = "./"
batch_size = 128
num_steps = 100

parser = argparse.ArgumentParser()
parser.add_argument('run_opt', type=int, default=1, help='An integer: 1 to train, 2 to test')
parser.add_argument('--data_path', type=str, default=data_path, help='The full path of the training data')
args = parser.parse_args()
if args.data_path:
    data_path = args.data_path





caseAttributeNameList = ['(case) AMOUNT_REQ']
activityAttributeList = ['concept:name','lifecycle:transition', 'Activity', 'Resource', 'Complete Timestamp']
defaultAtributeList = ['ID', 'Case ID']

useTimeAttributeList = ['Complete Timestamp']
useBooleanAttributeList = []
useFloatAttributeList = ['(case) AMOUNT_REQ']
useClassAttributeList = ['Activity', 'Resource']


caseActivityDict, vocabulary, timeOrderEventsArray, timeOrderLabelArray = load_data_from_db(num_steps = num_steps, 
        defaultAtributeList= defaultAtributeList, activityAttributeList = activityAttributeList , caseAttributeNameList = caseAttributeNameList, 
        useTimeAttributeList = useTimeAttributeList, useBooleanAttributeList = useBooleanAttributeList, useFloatAttributeList = useFloatAttributeList, useClassAttributeList = useClassAttributeList, 
        caseColumnName = "Case ID", timeColumnName = "Complete Timestamp", idColumnName = "ID", 
        dbName = "bpi2012.db", tableName = "bpi2012_new", timeStrp = "%Y-%m-%d %H:%M:%S")


reshapeX = np.reshape(timeOrderEventsArray, 
    ((timeOrderEventsArray.shape[0], timeOrderEventsArray.shape[1] * timeOrderEventsArray.shape[2])))

train_num = int(reshapeX.shape[0] * 0.9)
train_X = reshapeX[:train_num,:]
test_X = reshapeX[train_num:,:]
train_y = timeOrderLabelArray[:train_num]
test_y = timeOrderLabelArray[train_num:]

from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
rbm = BernoulliRBM(random_state=0, verbose=True)
svc = LinearSVC()
rbm_features_classifier = Pipeline(
    steps=[('rbm', rbm), ('svc', svc)])
rbm.learning_rate = 0.1
rbm.n_iter = 20
rbm.n_components = 50


if args.run_opt == 1:
    print('training...')
    clf = rbm_features_classifier.fit(train_X, train_y)
    print('training Complete')
    print(clf)

    from sklearn import metrics
    y_pred = clf.predict(test_X)
    print(metrics.classification_report(test_y, y_pred))
    print('accuracy_score:', metrics.accuracy_score(test_y, y_pred))

    print('rbm learning rate:', rbm.learning_rate)
    print('rbm n itern', rbm.n_iter)
    print('rbm n componets', rbm.n_components)
elif args.run_opt == 2:
    print("load model")
    # model = load_model("my_model_2012_minusstarttime.h5")

    # predictTrueNum = 0
    # predictFalseNum = 0
    # actualTrueNum = 0
    # actualFalseNum = 0
    # overTime = 30
    # predict_y = model.predict(test_X[0:200])
    # for i in range(predict_y.shape[0]):
        # print("actual:", test_y[i], " predict:", predict_y[i])

    print('done')







