
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
import keras_metrics


import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import argparse
import sqlite3
import datetime
import time
import json
import copy
from import_db import *
from import_txt import *

from classifier_helper import *

"""To run this code, you'll need to first download and extract the text dataset
    from here: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz. Change the
    data_path variable below to your local exraction path"""

data_path = "./"
batch_size = 128
num_steps = 100

use_dropout=True
epoch_num = 10
version = 0
# version 0: 256, 64 no dropout
# version 4: 256, 64 with dropout


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

train_num = int(timeOrderEventsArray.shape[0] * 0.9)

reshapeX = np.reshape(timeOrderEventsArray, 
    ((timeOrderEventsArray.shape[0], timeOrderEventsArray.shape[1] * timeOrderEventsArray.shape[2])))

train_num = int(reshapeX.shape[0] * 0.9)
train_X = reshapeX[:train_num,:]
test_X = reshapeX[train_num:,:]
train_y = timeOrderLabelArray[:train_num]
test_y = timeOrderLabelArray[train_num:]

# train_X = timeOrderEventsArray[:train_num,:,:]
# test_X = timeOrderEventsArray[train_num:,:,:]
# train_y = timeOrderLabelArray[:train_num]
# test_y = timeOrderLabelArray[train_num:]

print("begin model")

hidden_size = vocabulary
model = Sequential()
model.add(Masking(mask_value=0,input_shape=(num_steps * vocabulary,)))
model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

optimizer = Adam()

model.compile(loss=losses.binary_crossentropy, 
    optimizer='adam', 
    metrics=['binary_crossentropy','accuracy'])

print("end model")
print(model.summary())
checkpointer = ModelCheckpoint(filepath=data_path + '/bpi2012_mlp_{epoch:02d}_' + str(batch_size) + '_predict_overtime_v' + str(version) + '.hdf5', verbose=1)


if args.run_opt == 1:
    history = model.fit(train_X, train_y, epochs=epoch_num, batch_size=batch_size, validation_data=(test_X, test_y), verbose=1, shuffle=True, callbacks=[checkpointer])
    modelNameStr = "bpi2012_" + str(epoch_num) + "_" + str(batch_size) + "_predict_overtime_v" + str(version) + ".h5"
    model.save(modelNameStr)
    #model.save(data_path + "final_model.hdf5")

    print('final model predictions')
    predict_y = prob_to_class(model.predict(test_X))
    print(classification_report(test_y, predict_y))

    # for i in range(epoch_num):
    #     print('load model from: bpi2012_0' + str(i+1) + '_' + str(batch_size) + '_predict_overtime_v' + str(version) + '.hdf5')
    #     model = load_model(
    #         'bpi2012_0' + str(i+1) + '_' + str(batch_size) + '_predict_overtime_v' + str(version) + '.hdf5'
    #     )
    #     score, acc = model.evaluate(test_X, test_y, batch_size=batch_size)
    #     print('Test score:', score, 'Test accuracy', acc)
    #     predict_y = prob_to_class(model.predict(test_X))
    #     print(classification_report(test_y, predict_y))

elif args.run_opt == 2:
    for i in range(epoch_num):
        print('load model from: bpi2012_0' + str(i+1) + '_' + str(batch_size) + '_predict_overtime_v' + str(version) + '.hdf5')
        model = load_model(
            'bpi2012_0' + str(i+1) + '_' + str(batch_size) + '_predict_overtime_v' + str(version) + '.hdf5'
            # ,custom_objects={
            #     'precision':keras_metrics.precision(), 
            #     'recall': keras_metrics.recall()
            # }
        )
        predict_y = prob_to_class(model.predict(test_X))
        print(classification_report(test_y, predict_y))







