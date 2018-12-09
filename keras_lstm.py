
from __future__ import print_function
import collections
import os
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM, Masking
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras import regularizers
from keras import losses


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
import normal


from sklearn.neighbors import KNeighborsRegressor

"""To run this code, you'll need to first download and extract the text dataset
    from here: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz. Change the
    data_path variable below to your local exraction path"""

data_path = "./model/"
verbose = 1
batch_size = 128
num_steps = 100
l2_num = 0.1

use_dropout=False
epoch_num = 10
from_num = 5

parser = argparse.ArgumentParser()
# python3 keras_lstm.py 1 --batch_size 256 --l2 0.5 --from_num 5 --use_dropout True  --verbose 1 --num_steps 100 --epoch_num 10
parser.add_argument('run_opt', type=int, default=1, help='An integer: 1 to train, 2 to test')
parser.add_argument('--data_path', type=str, default=data_path, help='The full path of the training data')
parser.add_argument('--batch_size', type=int, default=batch_size, help='batch_size')
parser.add_argument('--num_steps', type=int, default=num_steps, help='num_steps')
parser.add_argument('--l2', type=float, default=l2_num, help='l2')
parser.add_argument('--use_dropout', type=bool, default=use_dropout, help='use_dropout')
parser.add_argument('--epoch_num', type=int, default=epoch_num, help='epoch_num')
parser.add_argument('--from_num', type=int, default=from_num, help='from_num')
parser.add_argument('--verbose', type=int, default=verbose, help='verbose')
args = parser.parse_args()
if args.data_path:
    data_path = args.data_path
    batch_size = args.batch_size
    num_steps = args.num_steps
    l2_num = args.l2
    use_dropout = args.use_dropout
    epoch_num = args.epoch_num
    from_num = args.from_num
    verbose = args.verbose

caseAttributeNameList = ['(case) AMOUNT_REQ']
activityAttributeList = ['concept:name','lifecycle:transition', 'Activity', 'Resource', 'Complete Timestamp']
defaultAtributeList = ['ID', 'Case ID']

useTimeAttributeList = ['Complete Timestamp']
useBooleanAttributeList = []
useFloatAttributeList = ['(case) AMOUNT_REQ']
useClassAttributeList = ['Activity', 'Resource']


caseActivityDict, vocabulary, timeOrderEventsArray, timeOrderLabelArray, timeOrderNowTimeArray = load_data_from_db(from_num = from_num, num_steps = num_steps, 
        defaultAtributeList= defaultAtributeList, activityAttributeList = activityAttributeList , caseAttributeNameList = caseAttributeNameList, 
        useTimeAttributeList = useTimeAttributeList, useBooleanAttributeList = useBooleanAttributeList, useFloatAttributeList = useFloatAttributeList, useClassAttributeList = useClassAttributeList, 
        caseColumnName = "Case ID", timeColumnName = "Complete Timestamp", idColumnName = "ID", 
        dbName = "bpi2012.db", tableName = "bpi2012_new", timeStrp = "%Y-%m-%d %H:%M:%S")



#caseActivityDict, vocabulary = load_data_from_db(defaultAtributeList, activityAttributeList, caseAttributeNameList, useTimeAttributeList, useBooleanAttributeList, useFloatAttributeList, useClassAttributeList, "case id")
#nowTimeDict = build_steps(caseActivityDict, 200, vocabulary)
#timeOrderEventsArray, timeOrderLabelArray = orderByTime(nowTimeDict, num_steps, vocabulary)
#print(caseActivityDict["5ceab127a2ec35a9"])


train_num = int(timeOrderEventsArray.shape[0] * 0.9)
test_num = int(timeOrderEventsArray.shape[0] * 0.9)
train_X = timeOrderEventsArray[:train_num,:,:]
test_X = timeOrderEventsArray[test_num:,:,:]
train_y = timeOrderLabelArray[:train_num]
test_y = timeOrderLabelArray[test_num:]
test_nowtime = timeOrderNowTimeArray[test_num:]

print("begin model")

# v3 增加正则化表达
if args.run_opt == 1:

    hidden_size = vocabulary
    model = Sequential()
    #model.add(Embedding(vocabulary, hidden_size, input_length=num_steps, mask_zero = True))
    model.add(Masking(mask_value=0,input_shape=(num_steps, vocabulary)))
    model.add(LSTM(hidden_size, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
    if use_dropout:
        model.add(Dropout(0.5))
    #model.add(Dense(hidden_size2))
    model.add(Activation('relu'))
    model.add(LSTM(hidden_size, return_sequences=False))
    if use_dropout:
        model.add(Dropout(0.5))
    #model.add(TimeDistributed(Dense(vocabulary)))
    #model.add(Dense(1))
    #model.add(Dense(1, kernel_regularizer=regularizers.l2(0.1), activity_regularizer=regularizers.l1(0.1)))
    model.add(Dense(1, kernel_regularizer=regularizers.l2(l2_num)))
    model.add(Activation('relu'))

    optimizer = Adam(clipvalue=0.5)
    #optimizer = Adam()
    model.compile(loss=losses.mean_squared_error, optimizer='adam', metrics=['MAE','MSE','MAPE','MSLE'])
    print("end model")

    print(model.summary())
    checkpointer = ModelCheckpoint(filepath=data_path + '/bpi2012_epoch_{epoch:02d}_batch_' + str(batch_size) + '_l2_' + str(l2_num) + '_from_num_' + str(from_num) + '_use_dropout_' + str(use_dropout) + '_predict_left_time_v3' + '.hdf5', verbose=verbose)



    #model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs,
    #                    validation_data=valid_data_generator.generate(),
    #                    validation_steps=len(valid_data)//(batch_size*num_steps), callbacks=[checkpointer])
    # model.fit_generator(train_data_generator.generate(), 2000, num_epochs,
    #                     validation_data=valid_data_generator.generate(),
    #                     validation_steps=10)
    print("epoch:", epoch_num, "batch_size:", batch_size, "l2", l2_num, "from_num", from_num, "use_dropout:", use_dropout)
    history = model.fit(train_X, train_y, epochs=epoch_num, batch_size=batch_size, validation_data=(test_X, test_y), verbose=verbose, shuffle=True, callbacks=[checkpointer])

    save_data_from_db(history.history, dirPath = data_path, description = 'history_bpi2012_batch_' + str(batch_size) + '_l2_' + str(l2_num) + '_from_num_' + str(from_num) + '_use_dropout_' + str(use_dropout), version = "3")

    #modelNameStr = "bpi2012_" + str(epoch_num) + "_" + str(batch_size) + "_" + "predict_left_timev3" + ".h5"
    #model.save('modelNameStr')
    #model.save(data_path + "final_model.hdf5")
elif args.run_opt == 2:
    print("load model")


    model = load_model("./model/bpi2012_epoch_05_batch_128_l2_0.01_from_num_5_use_dropout_True_predict_left_time_v3.hdf5")

    TP = 0
    FP = 0
    FN = 0
    TN = 0
    P = 0
    N = 0
    overTime = 30
    predict_y = model.predict(test_X[:],verbose = 1)
    for i in range(predict_y.shape[0]):
        if i % 10000 == 0:
            print("now processing", i)

        nowTime = test_nowtime[i]
        leftTime = test_y[i]
        actualAllTime = nowTime + leftTime
        predictAllTime = nowTime + predict_y[i][0]

        #print(nowTime, leftTime, actualAllTime, predictAllTime)
        #print("actual:", test_y[i], " predict:", predict_y[i])
        #input()

        if actualAllTime > overTime and predictAllTime > overTime:
            TN += 1
            N += 1
        elif actualAllTime < overTime and predictAllTime < overTime:
            TP += 1
            P += 1
        elif actualAllTime > overTime and predictAllTime < overTime:
            FP += 1
            N += 1
        else:
            FN += 1
            P += 1
    print("TP:",TP," FP:",FP, " FN:", FN," TN:",TN)
    print("positive accuracy:",(TP+TN)/float(P+N))
    #print("error rate:",(FP+FN)/float(P+N))
    print("positive recall:",(TP)/float(P))
    print("negative recall:",(TN)/float(N))
    print("positive precision:",(TP)/float(TP+FP))
    print("negative precision:",(TN)/float(TN+FN))
    print("positive f1:",2/(float(TP+FP)/TP + float(P)/TP))
    print("negative f1:",2/(float(TN+FN)/TN + float(N)/TN))




elif args.run_opt == 3:
    print("knn")
    neigh = KNeighborsRegressor(n_neighbors=2, weights='distance', metric='manhattan')
    train_X = train_X.reshape((train_X.shape[0],train_X.shape[1]*train_X.shape[2]))
    #train_X = train_X[:10000]
    #train_y = train_y[:10000]
    test_X = test_X.reshape((test_X.shape[0],test_X.shape[1]*test_X.shape[2]))
    print(train_X.shape, test_X.shape)
    neigh.fit(train_X, train_y)
    print("finish fit")
    predictError = 0
    for i in range(test_X.shape[0]):
        predict_y = neigh.predict([test_X[i]])
        #print("actual:", test_y[i], " predict:", predict_y)
        predictError += abs(predict_y[0] - test_y[i])
    print("mae:", predictError / float(test_X.shape[0]))


elif args.run_opt == 4:
    print("GradientBoostingRegressor")
    import numpy as np
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.datasets import make_friedman1
    from sklearn.ensemble import GradientBoostingRegressor

    train_X = train_X.reshape((train_X.shape[0],train_X.shape[1]*train_X.shape[2]))
    train_X = train_X[:10000]
    train_y = train_y[:10000]
    test_X = test_X.reshape((test_X.shape[0],test_X.shape[1]*test_X.shape[2]))
    est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls').fit(train_X, train_y)
    print("mse:", mean_squared_error(test_y, est.predict(test_X)))
    print("mae:", mean_absolute_error(test_y, est.predict(test_X)))
