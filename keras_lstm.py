
from __future__ import print_function
import collections
import os
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
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

"""To run this code, you'll need to first download and extract the text dataset
    from here: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz. Change the
    data_path variable below to your local exraction path"""

data_path = "./"
batch_size = 128
num_epochs = 5
num_steps = 200
max_timestamp = 1516449782
min_timestamp = 1399068000

hidden_size = 500
use_dropout=True
epoch_num = 5

parser = argparse.ArgumentParser()
parser.add_argument('run_opt', type=int, default=1, help='An integer: 1 to train, 2 to test')
parser.add_argument('--data_path', type=str, default=data_path, help='The full path of the training data')
args = parser.parse_args()
if args.data_path:
    data_path = args.data_path

caseAttributeNameList = ['(case) amount_applied0','(case) amount_applied1','(case) amount_applied2','(case) amount_applied3','(case) applicant','(case) application','(case) area','(case) basic payment','(case) cross_compliance','(case) department','(case) greening','(case) number_parcels','(case) payment_actual0','(case) payment_actual1','(case) payment_actual2','(case) payment_actual3','(case) penalty_ABP','(case) penalty_AGP','(case) penalty_AJLP','(case) penalty_AUVP','(case) penalty_AVBP','(case) penalty_AVGP','(case) penalty_AVJLP','(case) penalty_AVUVP','(case) penalty_B16','(case) penalty_B2','(case) penalty_B3','(case) penalty_B4','(case) penalty_B5','(case) penalty_B5F','(case) penalty_B6','(case) penalty_BGK','(case) penalty_BGKV','(case) penalty_BGP','(case) penalty_C16','(case) penalty_C4','(case) penalty_C9','(case) penalty_CC','(case) penalty_GP1','(case) penalty_JLP1','(case) penalty_JLP2','(case) penalty_JLP3','(case) penalty_JLP5','(case) penalty_JLP6','(case) penalty_JLP7','(case) penalty_V5','(case) penalty_amount0','(case) penalty_amount1','(case) penalty_amount2','(case) penalty_amount3','(case) program-id','(case) redistribution','(case) rejected','(case) risk_factor','(case) selected_manually','(case) selected_random','(case) selected_risk','(case) small farmer','(case) year','(case) young farmer']
activityAttributeList = ['doctype','note','subprocess','success', "concept:name"]
defaultAtributeList = ['case id', 'Activity', 'Complete Timestamp', 'Resource']

useTimeAttributeList = ['Complete Timestamp']
useBooleanAttributeList = ['(case) basic payment', '(case) greening','(case) penalty_ABP','(case) penalty_AGP','(case) penalty_AJLP','(case) penalty_AUVP','(case) penalty_AVBP','(case) penalty_AVGP','(case) penalty_AVJLP','(case) penalty_AVUVP','(case) penalty_B16','(case) penalty_B2','(case) penalty_B3','(case) penalty_B4','(case) penalty_B5','(case) penalty_B5F','(case) penalty_B6','(case) penalty_BGK','(case) penalty_BGKV','(case) penalty_BGP','(case) penalty_C16','(case) penalty_C4','(case) penalty_C9','(case) penalty_CC','(case) penalty_GP1','(case) penalty_JLP1','(case) penalty_JLP2','(case) penalty_JLP3','(case) penalty_JLP5','(case) penalty_JLP6','(case) penalty_JLP7','(case) penalty_V5','(case) redistribution', '(case) rejected','(case) selected_manually','(case) selected_random', '(case) selected_risk', '(case) small farmer', '(case) young farmer']
useFloatAttributeList = ['(case) amount_applied0', '(case) amount_applied1', '(case) amount_applied2', '(case) amount_applied3', '(case) area', '(case) cross_compliance', '(case) number_parcels', '(case) payment_actual0','(case) payment_actual1','(case) payment_actual2','(case) payment_actual3','(case) penalty_amount0','(case) penalty_amount1','(case) penalty_amount2','(case) penalty_amount3', '(case) risk_factor']
useClassAttributeList = ['Activity', '(case) department', '(case) year', 'Resource']





def build_steps(caseActivityDict, num_steps, vocabulary):
    nowTimeDict = {}
    skipNum = 0
    for caseId, eventList in caseActivityDict.items():
        nowEventsArray = np.array([])
        if len(eventList) > num_steps:
            skipNum +=1
            continue
        eventId = 0
        for eventTuple in eventList:
            if eventId == len(eventList) - 1 :
                break
            featureArray = eventTuple[0]
            endTimeStamp = eventTuple[1]
            if nowEventsArray.shape[0] == 0:
                nowEventsArray = np.array([featureArray[:]])
            else:
                nowEventsArray = np.row_stack((nowEventsArray, featureArray))
            #print(nowEventsArray)
            #print(nowEventsArray.shape)
            dataAndLabelDict = {}
            xFeature = np.zeros((num_steps, vocabulary))
            xFeature[ : nowEventsArray.shape[0], : ] = nowEventsArray[:,:]
            dataAndLabelDict["feature"] = xFeature
            nextEndTime = eventList[eventId + 1][1]
            dataAndLabelDict["label"] = (nextEndTime - min_timestamp) / float(max_timestamp - min_timestamp)
            #print(dataAndLabelDict["label"])
            eventIdStr = "%05d" % eventId
            nowTimeDict[str(endTimeStamp) + "_" + caseId + "_" + eventIdStr ] = dataAndLabelDict
            #print(nowTimeDict)
            #input()
            eventId += 1

    print("skipNum:", skipNum)
    return nowTimeDict

def orderByTime(nowTimeDict, num_steps, vocabulary):
    timeOrderEventsArray = np.zeros((len(nowTimeDict), num_steps, vocabulary))
    timeOrderLabelArray = np.zeros((len(nowTimeDict)))
    keys = sorted(nowTimeDict.keys())
    keyIndex = 0
    for key in keys:
        featureArray = nowTimeDict[key]["feature"]
        timeOrderEventsArray[keyIndex, :, :] = featureArray
        timeOrderLabelArray[keyIndex] = nowTimeDict[key]["label"]
        if (keyIndex % 10000 == 0 or keyIndex > 47000):
            print("orderByTime:", keyIndex)
        keyIndex += 1
        #print(timeOrderEventsArray)
        #print(timeOrderLabelArray)
        #input()
    return timeOrderEventsArray, timeOrderLabelArray



caseActivityDict, vocabulary, timeOrderEventsArray, timeOrderLabelArray = load_data_from_db(limitNum = 30000, offsetNum = 0, num_steps = num_steps, defaultAtributeList= defaultAtributeList, activityAttributeList = activityAttributeList
        , caseAttributeNameList = caseAttributeNameList, useTimeAttributeList = useTimeAttributeList, useBooleanAttributeList = useBooleanAttributeList, useFloatAttributeList = useFloatAttributeList, useClassAttributeList = useClassAttributeList, caseColumnName = "case id", timeColumnName = "Complete Timestamp", idColumnName = "ID", dbName = "bpi2018.db", tableName = "BPIC2018_new", timeStrp = "%Y/%m/%d %H:%M:%S")


#caseActivityDict, vocabulary = load_data_from_db(defaultAtributeList, activityAttributeList, caseAttributeNameList, useTimeAttributeList, useBooleanAttributeList, useFloatAttributeList, useClassAttributeList, "case id")
#nowTimeDict = build_steps(caseActivityDict, 200, vocabulary)
#timeOrderEventsArray, timeOrderLabelArray = orderByTime(nowTimeDict, num_steps, vocabulary)
#print(caseActivityDict["5ceab127a2ec35a9"])


train_num = int(timeOrderEventsArray.shape[0] * 0.8)
train_X = timeOrderEventsArray[:train_num,:,:]
test_X = timeOrderEventsArray[train_num:,:,:]
train_y = timeOrderEventsArray[:train_num]
test_y = timeOrderEventsArray[train_num:]

print("begin model")

#hidden_size = vocabulary
model = Sequential()
#model.add(Embedding(vocabulary, hidden_size, input_length=num_steps, mask_zero = True))
model.add(LSTM(hidden_size, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(LSTM(hidden_size, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
if use_dropout:
    model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(vocabulary)))
model.add(Activation('softmax'))

optimizer = Adam()
model.compile(loss=losses.mean_absolute_error, optimizer='adam', metrics=['MAE'])
print("end model")

print(model.summary())
#checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)


if args.run_opt == 1:
    #model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs,
    #                    validation_data=valid_data_generator.generate(),
    #                    validation_steps=len(valid_data)//(batch_size*num_steps), callbacks=[checkpointer])
    # model.fit_generator(train_data_generator.generate(), 2000, num_epochs,
    #                     validation_data=valid_data_generator.generate(),
    #                     validation_steps=10)
    history = model.fit(train_X, train_y, epochs=epoch_num, batch_size=batch_size, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    #model.save(data_path + "final_model.hdf5")





