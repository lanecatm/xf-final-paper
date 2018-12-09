
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
from classifier_helper import *

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
# useClassAttributeList = ['Activity', 'Resource']
useClassAttributeList = ['Activity']


caseActivityDict, vocabulary, timeOrderEventsArray, timeOrderLabelArray = load_data_from_db(num_steps = num_steps, 
        defaultAtributeList= defaultAtributeList, activityAttributeList = activityAttributeList , caseAttributeNameList = caseAttributeNameList, 
        useTimeAttributeList = useTimeAttributeList, useBooleanAttributeList = useBooleanAttributeList, useFloatAttributeList = useFloatAttributeList, useClassAttributeList = useClassAttributeList, 
        caseColumnName = "Case ID", timeColumnName = "Complete Timestamp", idColumnName = "ID", 
        dbName = "bpi2012.db", tableName = "bpi2012_new", timeStrp = "%Y-%m-%d %H:%M:%S")


reshapeX = np.reshape(timeOrderEventsArray, 
    ((timeOrderEventsArray.shape[0], timeOrderEventsArray.shape[1] * timeOrderEventsArray.shape[2])))

train_num = int(reshapeX.shape[0] * 0.9)



import gc
gc.collect()

from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn import metrics



if args.run_opt == 1:
    par_num = 50
    train_X_split = np.array_split(reshapeX[:train_num,:], par_num)
    train_y_split = np.array_split(timeOrderLabelArray[:train_num], par_num)
    test_X_split = np.array_split(reshapeX[train_num:,:],par_num)
    test_y_split = np.array_split(timeOrderLabelArray[train_num:],par_num)

    gc.collect()

    rbm = BernoulliRBM(random_state=0, verbose=True)

    rbm.learning_rate = 0.1
    rbm.n_iter = 5
    rbm.n_components = 10
    
    print('Training RBM ...')
    for i, chunk in enumerate(train_X_split):
        print('rbm.partial_fit chunk ' + str(i+1) + '/' + str(par_num) + '\r', end="")
        rbm.partial_fit(chunk)
    print('Training RBM Complete')
    gc.collect()

    sgdc = SGDClassifier()
    print('Training Classifier ...')
    for i in range(par_num):
        print('sgdc.partial_fit chunk ' + str(i+1) + '/' + str(par_num) + '\r', end="")
        sgdc.partial_fit(rbm.transform(train_X_split[i]), train_y_split[i], classes=[0,1])
    clf = sgdc
    # clf = sgdc.fit(rbm.transform(train_X), train_y)
    # clf = rbm_features_classifier.fit(train_X, train_y)
    print('Training Classifier Complete')
    print(clf)

    # test
    tnTotal, fpTotal, fnTotal, tpTotal = (0,0,0,0)
    for i in range(par_num):
        tn, fp, fn, tp = metrics.confusion_matrix(
            test_y_split[i], 
            clf.predict(rbm.transform(test_X_split[i])), 
            [0, 1]).ravel()
        tnTotal += tn
        fpTotal += fp
        fnTotal += fn
        tpTotal += tp

    precision, recall, f1, accuracy = toScore(tnTotal, fpTotal, fnTotal, tpTotal)

    print('precision', precision)
    print('recall', recall)
    print('f1-score', f1)
    print('accuracy', accuracy)

    print('rbm learning rate:', rbm.learning_rate)
    print('rbm n itern', rbm.n_iter)
    print('rbm n componets', rbm.n_components)
elif args.run_opt == 2:
    print('2')
    train_X = reshapeX[:train_num,:]
    test_X = reshapeX[train_num:,:]
    train_y = timeOrderLabelArray[:train_num]
    test_y = timeOrderLabelArray[train_num:]
    gc.collect()


    rbm = BernoulliRBM(random_state=0, verbose=True)
    rbm.learning_rate = 0.1
    rbm.n_iter = 5
    rbm.n_components = 10
    svc = LinearSVC()
    rbm_features_classifier = Pipeline(steps=[('rbm', rbm), ('svc', svc)])
    print('Training ...')
    clf = rbm_features_classifier.fit(train_X, train_y)
    print(clf)

    print('Testing ...')
    y_pred = clf.predict(test_X)
    print(metrics.classification_report(test_y, y_pred))
    print('accuracy_score:', metrics.accuracy_score(test_y, y_pred))
    
    print('\nrbm learning rate:', rbm.learning_rate)
    print('rbm n itern', rbm.n_iter)
    print('rbm n componets', rbm.n_components)
elif args.run_opt == 3:
    print('scan posibilities with whole data')
    # train_X = reshapeX[:train_num,:]
    # test_X = reshapeX[train_num:,:]
    # train_y = timeOrderLabelArray[:train_num]
    # test_y = timeOrderLabelArray[train_num:]
    # gc.collect()
    # # scan for best 
    # hiddens = [300, 100, 50]
    # iters = [50, 100, 800]
    # algos = ['RandForest', 'LinearSVC', 'logistic']
    # for hidden in hiddens:
    #     for itern in iters:
elif args.run_opt == 4:
    par_num = 50
    train_X_split = np.array_split(reshapeX[:train_num,:], par_num)
    train_y_split = np.array_split(timeOrderLabelArray[:train_num], par_num)
    test_X_split = np.array_split(reshapeX[train_num:,:], par_num)
    test_y_split = np.array_split(timeOrderLabelArray[train_num:], par_num)

    hiddens = [1000, 300, 100]
    iters = [100, 1000, 5000]
    for hidden in hiddens:
        for itern in iters:
            
            rbm = BernoulliRBM(random_state=0, verbose=True)
            rbm.learning_rate = 0.1
            rbm.n_iter = itern
            rbm.n_components = hidden

            print('Training RBM ... with hidden componets:' + str(hidden) + '\t iteration number:', itern)
            for i, chunk in enumerate(train_X_split):
                print('rbm.partial_fit chunk ' + str(i+1) + '/' + str(par_num) + '\r', end="")
                rbm.partial_fit(chunk)
            print('Training RBM Complete')

            sgdc = SGDClassifier()
            print('Training Classifier ...')
            for i in range(par_num):
                print('sgdc.partial_fit chunk ' + str(i+1) + '/' + str(par_num) + '\r', end="")
                sgdc.partial_fit(rbm.transform(train_X_split[i]), train_y_split[i], classes=[0,1])
            clf = sgdc
            print('Training Classifier Complete')
            # print(clf)

            # test
            tnTotal, fpTotal, fnTotal, tpTotal = (0,0,0,0)
            for i in range(par_num):
                tn, fp, fn, tp = metrics.confusion_matrix(
                    test_y_split[i], 
                    clf.predict(rbm.transform(test_X_split[i])), 
                    [0, 1]).ravel()
                tnTotal += tn
                fpTotal += fp
                fnTotal += fn
                tpTotal += tp

            precision, recall, f1, accuracy = toScore(tnTotal, fpTotal, fnTotal, tpTotal)

            print('\n----------------------------')
            print('tn, fp, fn, tp   ', tnTotal, fpTotal, fnTotal, tpTotal)
            print('hidden componets:', hidden, '\t iteration number:', itern)
            print('precision:', precision)
            print('recall:', recall)
            print('f1-score:', f1)
            print('accuracy:', accuracy)
            print('----------------------------\n')
            gc.collect()



















