from keras.utils import to_categorical
from keras.utils import np_utils
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import argparse
import sqlite3
import datetime
import time
import json
import copy
from import_txt import *



def build_min_and_max_timestamp(attributeName, dbName, tableName, timeStrp):
    conn = sqlite3.connect(dbName)
    c = conn.cursor()
    print("connect db successfully")
    sql = "SELECT min([" + attributeName + "]), max([" + attributeName + "]) from " + tableName
    print(sql)
    cursor = c.execute(sql)
    for row in cursor:
        minValue = row[0] 
        maxValue = row[1]
        
    conn.close()
    try:
        minTimeStr = minValue.split(".")[0]
        minTime = datetime.datetime.strptime(minTimeStr, timeStrp)
        minTimeStamp = int(time.mktime(minTime.timetuple()))
        maxTimeStr = maxValue.split(".")[0]
        maxTime = datetime.datetime.strptime(maxTimeStr, timeStrp)
        maxTimeStamp = int(time.mktime(maxTime.timetuple()))
    except ValueError:
        print("minTimeStr",minTimeStr, " maxTimeStr",maxTimeStr)
    return minTimeStamp, maxTimeStamp
    

def build_min_and_max_value(attributeName, dbName, tableName):
    conn = sqlite3.connect(dbName)
    c = conn.cursor()
    print("connect db successfully")
    cursor = c.execute("SELECT min( CAST( [" + attributeName + "] as REAL)), max( CAST( [" + attributeName + "] as REAL)) from " + tableName)
    for row in cursor:
        minValue = row[0] 
        maxValue = row[1]
    conn.close()
    if minValue == "":
        minValue = 0
    else:
        minValue = float(minValue)
    if maxValue == "":
        maxValue = 0
    else:
        maxValue = float(maxValue)
    return minValue, maxValue



def build_attribute_to_id_dict(attributeName, dbName, tableName):
    conn = sqlite3.connect(dbName)
    c = conn.cursor()
    print("connect db successfully")
    cursor = c.execute("SELECT distinct([" + attributeName + "])  from " + tableName)
    activityNameList = []
    for row in cursor:
        activityNameList.append(row[0])
    conn.close()
    activity_to_id = dict(zip(activityNameList, range(len(activityNameList))))
    print(activity_to_id)
    return activity_to_id


def attributeSqlStr(targetList):
    attrName2IdDict = {}
    id2attrNameDict = {}
    ansStr = "[id]"
    index = 0
    attrName2IdDict["id"] = index
    id2attrNameDict[index] = "id"
    index += 1
    for attr in targetList:
        ansStr = ansStr +  ", [" + attr + "]"
        attrName2IdDict[attr] = index
        id2attrNameDict[index] = attr
        index += 1
    return ansStr, attrName2IdDict, id2attrNameDict

def change_row_into_array(row, attrName2IdDict, useTimeAttributeList, useBooleanAttributeList, useFloatAttributeList, useClassAttributeList, timeStrp, classAttributeToIdDict, minValueDict, maxValueDict, min_timestamp, max_timestamp ):

    # change class
    classAttributeList = np.array([])
    for attr in useClassAttributeList:
        columnId = attrName2IdDict[attr]
        attrStr = row[columnId]
        #print(attr, ":", attrStr)
        attrValue = classAttributeToIdDict[attr][attrStr]
        oneHotAttrValue = np_utils.to_categorical([attrValue],num_classes=len(classAttributeToIdDict[attr]))
        classAttributeList = np.append(classAttributeList, oneHotAttrValue[0])
    #print(classAttributeList)
    #input("next:")

    # change time
    timeAttributeList = []
    for attr in useTimeAttributeList:
        endTimeStr = row[attrName2IdDict[attr]].split(".")[0]
        endTime = datetime.datetime.strptime(endTimeStr, timeStrp)
        endTimeStamp = int(time.mktime(endTime.timetuple()))
        endTimeStampNormal = (endTimeStamp - min_timestamp) / float(max_timestamp - min_timestamp)
        timeAttributeList.append(endTimeStampNormal)

    # change boolean
    booleanAttributeList = []
    for attr in useBooleanAttributeList:
        columnId = attrName2IdDict[attr]
        attrValue = row[columnId]
        if attrValue == "true":
            booleanAttributeList.append(1)
        elif attrValue == "false":
            booleanAttributeList.append(0)
        else:
            booleanAttributeList.append(0)
            print("error boolean value:", attrValue)

    # change float
    floatAttributeList = []
    for attr in useFloatAttributeList:
        columnId = attrName2IdDict[attr]
        attrValue = row[columnId]
        try:
            floatAttrValue = float(attrValue)
            floatAttrValueNormal = (float(attrValue) - minValueDict[attr]) / (maxValueDict[attr] - minValueDict[attr])
            floatAttributeList.append(floatAttrValueNormal)
        except ValueError:
            if attrValue != '':
                print("float value error:", attrValue)
            floatAttributeList.append(0)
    featureArray = np.append(np.append(classAttributeList, timeAttributeList), np.append(booleanAttributeList, floatAttributeList))
    return featureArray, endTimeStamp

def get_event_number(dbName, tableName):
    conn = sqlite3.connect(dbName)
    c = conn.cursor()
    print("connect db successfully")

    sql = "SELECT COUNT(*) FROM " + tableName;
    print(sql)

    cursor = c.execute(sql)
    for row in cursor:
        print("event number: ",  row[0])
        return row[0]


def load_basic_data_from_db(dbName, tableName, useClassAttributeList, useFloatAttributeList, filePath = "default"):
    if filePath == "default":
        filePath = "./" + dbName + ".txt"
    classAttributeToIdDict = {}
    for attr in useClassAttributeList:
        classAttributeToIdDict[attr] = build_attribute_to_id_dict(attributeName = attr, dbName = dbName, tableName = tableName)
    minValueDict = {}
    maxValueDict = {}
    for attr in useFloatAttributeList:
        minValue, maxValue = build_min_and_max_value(attributeName = attr, dbName = dbName, tableName = tableName)
        minValueDict[attr] = minValue
        maxValueDict[attr] = maxValue
    save_data_from_db([classAttributeToIdDict,minValueDict,maxValueDict],filePath)
    return classAttributeToIdDict,minValueDict,maxValueDict


def load_data_from_db(num_steps, defaultAtributeList, activityAttributeList, caseAttributeNameList, useTimeAttributeList, useBooleanAttributeList, useFloatAttributeList, useClassAttributeList, caseColumnName, timeColumnName, idColumnName, dbName, tableName, timeStrp):

    #classAttributeToIdDict = load_same_data_from_db()
    classAttributeToIdDict, minValueDict, maxValueDict  = load_basic_data_from_db(dbName, tableName, useClassAttributeList, useFloatAttributeList)
    min_timestamp, max_timestamp = build_min_and_max_timestamp(useTimeAttributeList[0], dbName, tableName, timeStrp)
    #classAttributeToIdDict, minValueDict, maxValueDict  = load_same_data_from_txt(dbName + ".txt")

    targetList = defaultAtributeList + activityAttributeList + caseAttributeNameList
    ansStr, attrName2IdDict, id2attrNameDict = attributeSqlStr(targetList)
    
    eventNumber = get_event_number(dbName, tableName)

    conn = sqlite3.connect(dbName)
    c = conn.cursor()
    print("connect db successfully")

    sql = "SELECT " + ansStr + " from " + tableName + " order by [" +  timeColumnName + "], [" + caseColumnName + "], [" + idColumnName + "]"
    print("sql:",sql)
    cursor = c.execute(sql)
    
    caseActivityDict = {}
    
    vocabulary = len(useTimeAttributeList) + len(useBooleanAttributeList) + len(useFloatAttributeList)
    for attr in useClassAttributeList:
        vocabulary += len(classAttributeToIdDict[attr])
    print("vocabulary", vocabulary)

    timeOrderEventsArray = np.zeros((eventNumber, num_steps, vocabulary))
    timeOrderLabelArray = np.zeros((eventNumber))

    rowIndex = 1
    timeOrderEventsArrayIndex = 0
    overNumStepEventNumber = 0
    for row in cursor:
        if (rowIndex % 10000 == 0):
            print("input ", rowIndex)
            #print("input row:", row)
        rowIndex += 1


        caseName = row[attrName2IdDict[caseColumnName]]
        featureArray, featureTimeStamp  = change_row_into_array(row, attrName2IdDict, useTimeAttributeList, useBooleanAttributeList, useFloatAttributeList, useClassAttributeList, timeStrp, classAttributeToIdDict, minValueDict, maxValueDict, min_timestamp, max_timestamp )

        if not caseName in caseActivityDict:
            caseActivityDict[caseName] = []
        else:
            # add event and label to list
            if len(caseActivityDict[caseName]) > num_steps:
                overNumStepEventNumber += 1
            else:
                # TODO check
                nowFeatureArray = np.array(caseActivityDict[caseName])

                timeOrderEventsArray[timeOrderEventsArrayIndex, : nowFeatureArray.shape[0], : ] = nowFeatureArray[:,:]
                timeOrderLabelArray[timeOrderEventsArrayIndex] = (featureTimeStamp - min_timestamp) / float(max_timestamp - min_timestamp)
                timeOrderEventsArrayIndex += 1

        caseActivityDict[caseName].append(featureArray)
    conn.close()

    print(useClassAttributeList + useTimeAttributeList + useBooleanAttributeList + useFloatAttributeList)
    #for eventList in caseActivityDict["5ceab127a2ec35a9"]:
    #    print(eventList)
    for feature in caseActivityDict[caseName]:
        print(feature)
    print(len(caseActivityDict[caseName]))

    print("overNumStepEventNumber:", overNumStepEventNumber)

    # TODO delete final zeros in timeOrderEventsArray and timeOrderLabelArray

    return caseActivityDict, vocabulary, timeOrderEventsArray, timeOrderLabelArray

