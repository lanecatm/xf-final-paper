
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
num_steps = 100

use_dropout=True
epoch_num = 2

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



#caseActivityDict, vocabulary = load_data_from_db(defaultAtributeList, activityAttributeList, caseAttributeNameList, useTimeAttributeList, useBooleanAttributeList, useFloatAttributeList, useClassAttributeList, "case id")
#nowTimeDict = build_steps(caseActivityDict, 200, vocabulary)
#timeOrderEventsArray, timeOrderLabelArray = orderByTime(nowTimeDict, num_steps, vocabulary)
#print(caseActivityDict["5ceab127a2ec35a9"])


train_num = int(timeOrderEventsArray.shape[0] * 0.8)
train_X = timeOrderEventsArray[:train_num,:,:]
test_X = timeOrderEventsArray[train_num:,:,:]
train_y = timeOrderLabelArray[:train_num]
test_y = timeOrderLabelArray[train_num:]

print("begin model")

hidden_size = vocabulary
model = Sequential()
#model.add(Embedding(vocabulary, hidden_size, input_length=num_steps, mask_zero = True))
model.add(Masking(mask_value=0,input_shape=(num_steps, vocabulary)))
model.add(LSTM(hidden_size, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
if use_dropout:
    model.add(Dropout(0.5))
#model.add(Dense(hidden_size2))
#model.add(Activation('relu'))
model.add(LSTM(hidden_size, return_sequences=False))
if use_dropout:
    model.add(Dropout(0.5))
#model.add(TimeDistributed(Dense(vocabulary)))
model.add(Dense(1))
model.add(Activation('linear'))

optimizer = Adam()
model.compile(loss=losses.mean_squared_error, optimizer='adam', metrics=['MAE','MSE','MAPE','MSLE'])
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
    history = model.fit(train_X, train_y, epochs=epoch_num, batch_size=batch_size, validation_data=(test_X, test_y), verbose=1, shuffle=True)

    model.save('my_model.h5')
    #model.save(data_path + "final_model.hdf5")





