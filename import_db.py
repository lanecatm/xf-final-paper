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



startCalculateEventNum = 1


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

def find_case_end_time(caseColumnName, useTimeAttributeList, dbName, tableName, timeStrp):
    caseEndTimeDict = {}
    conn = sqlite3.connect(dbName)
    c = conn.cursor()
    #print("connect db successfully")
    sql = "SELECT [" + caseColumnName + "], MAX([" + useTimeAttributeList[0] + "]) FROM " + tableName + " group by [" + caseColumnName +"]"
    print(sql)
    cursor = c.execute(sql)
    for row in cursor:
        caseId = row[0]
        maxValue = row[1]
        maxTimeStr = maxValue.split(".")[0]
        maxTime = datetime.datetime.strptime(maxTimeStr, timeStrp)
        maxTimeStamp = int(time.mktime(maxTime.timetuple()))
        caseEndTimeDict[caseId] = maxTimeStamp
    conn.close()
    #print("maxTimeStamp",maxTimeStamp)
    print(caseEndTimeDict[caseId])
    return caseEndTimeDict

def find_case_start_time(caseColumnName, useTimeAttributeList, dbName, tableName, timeStrp):
    caseEndTimeDict = {}
    conn = sqlite3.connect(dbName)
    c = conn.cursor()
    #print("connect db successfully")
    sql = "SELECT [" + caseColumnName + "], MIN([" + useTimeAttributeList[0] + "]) FROM " + tableName + " group by [" + caseColumnName +"]"
    print(sql)
    cursor = c.execute(sql)
    for row in cursor:
        caseId = row[0]
        maxValue = row[1]
        maxTimeStr = maxValue.split(".")[0]
        maxTime = datetime.datetime.strptime(maxTimeStr, timeStrp)
        maxTimeStamp = int(time.mktime(maxTime.timetuple()))
        caseEndTimeDict[caseId] = maxTimeStamp
    conn.close()
    #print("maxTimeStamp",maxTimeStamp)
    print(caseEndTimeDict[caseId])
    return caseEndTimeDict



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

def timeStrToTimeStamp(timeOriginStr, timeStrp):
    endTimeStr = timeOriginStr.split(".")[0]
    endTime = datetime.datetime.strptime(endTimeStr, timeStrp)
    endTimeStamp = int(time.mktime(endTime.timetuple()))
    return endTimeStamp

def change_row_into_array(row, attrName2IdDict, useTimeAttributeList, useBooleanAttributeList, useFloatAttributeList, useClassAttributeList, timeStrp, classAttributeToIdDict, minValueDict, maxValueDict, caseStartTime, timeUnit):

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
        endTimeStamp = timeStrToTimeStamp( row[attrName2IdDict[attr]], timeStrp)
        endTimeStampNormal = (endTimeStamp - caseStartTime) / float(timeUnit)
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
    save_data_from_db([classAttributeToIdDict,minValueDict,maxValueDict], description = dbName + "_basic_data")
    return classAttributeToIdDict,minValueDict,maxValueDict


def load_data_from_db(num_steps, defaultAtributeList, activityAttributeList, caseAttributeNameList, useTimeAttributeList, useBooleanAttributeList, useFloatAttributeList, useClassAttributeList, caseColumnName, timeColumnName, idColumnName, dbName, tableName, timeStrp):

    #classAttributeToIdDict = load_same_data_from_db()
    classAttributeToIdDict, minValueDict, maxValueDict  = load_basic_data_from_db(dbName, tableName, useClassAttributeList, useFloatAttributeList)
    min_timestamp, max_timestamp = build_min_and_max_timestamp(useTimeAttributeList[0], dbName, tableName, timeStrp)
    #classAttributeToIdDict, minValueDict, maxValueDict  = load_same_data_from_txt(description = dbName + "_basic_data")

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
    caseTimeDict = {}
    caseEndTimeDict = find_case_end_time(caseColumnName, useTimeAttributeList, dbName, tableName, timeStrp)
    caseStartTimeDict = find_case_start_time(caseColumnName, useTimeAttributeList, dbName, tableName, timeStrp)
    
    vocabulary = len(useTimeAttributeList) + len(useBooleanAttributeList) + len(useFloatAttributeList)
    for attr in useClassAttributeList:
        vocabulary += len(classAttributeToIdDict[attr])
    print("vocabulary", vocabulary)

    timeOrderEventsArray = np.zeros((eventNumber, num_steps, vocabulary))
    timeOrderLabelArray = np.zeros((eventNumber))

    rowIndex = 1
    timeOrderEventsArrayIndex = 0
    overNumStepEventNumber = 0
    lessNumEventNumber = 0
    
    trueNum = 0
    falseNum = 0
    for row in cursor:
        if (rowIndex % 10000 == 0):
            print("input ", rowIndex)
            #print("input row:", row)
        rowIndex += 1


        caseName = row[attrName2IdDict[caseColumnName]]

        featureArray, featureTimeStamp  = change_row_into_array(row, attrName2IdDict, useTimeAttributeList, useBooleanAttributeList, useFloatAttributeList, useClassAttributeList, timeStrp, classAttributeToIdDict, minValueDict, maxValueDict, caseStartTimeDict[caseName],8640000)
        if not caseName in caseActivityDict:
            caseActivityDict[caseName] = []
            caseTimeDict[caseName] = []
        else:
            # add event and label to list
            if len(caseActivityDict[caseName]) > num_steps:
                overNumStepEventNumber += 1
            elif len(caseActivityDict[caseName]) < startCalculateEventNum:
                lessNumEventNumber += 1
            else:
                # TODO check
                nowFeatureArray = np.array(caseActivityDict[caseName])
                nowFeatureTimeStamp = caseTimeDict[caseName][-1]

                timeOrderEventsArray[timeOrderEventsArrayIndex, : nowFeatureArray.shape[0], : ] = nowFeatureArray[:,:]
                caseEndTimeStamp = caseEndTimeDict[caseName]
                #if caseEndTimeStamp < featureTimeStamp:
                #    print("case end time error")
                # yik label
                caseDuration = caseEndTimeStamp - caseStartTimeDict[caseName]
                if caseDuration > 60 * 60 * 24 * 30:
                    timeOrderLabelArray[timeOrderEventsArrayIndex] = 1 
                    trueNum += 1
                else:
                    timeOrderLabelArray[timeOrderEventsArrayIndex] = 0
                    falseNum += 1
                #timeOrderLabelArray[timeOrderEventsArrayIndex] = (caseEndTimeStamp - nowFeatureTimeStamp) / 8640000.0
                #timeOrderLabelArray[timeOrderEventsArrayIndex] = (featureTimeStamp - min_timestamp) / float(60*60)
                #print(timeOrderLabelArray[timeOrderEventsArrayIndex])
                #input()
                timeOrderEventsArrayIndex += 1

        caseActivityDict[caseName].append(featureArray)
        caseTimeDict[caseName].append(featureTimeStamp)
    conn.close()

    print(useClassAttributeList + useTimeAttributeList + useBooleanAttributeList + useFloatAttributeList)
    #for eventList in caseActivityDict["5ceab127a2ec35a9"]:
    #    print(eventList)
    # for feature in caseActivityDict[caseName]:
    #     print(feature)
    print('len(caseActivityDict[caseName]):', len(caseActivityDict[caseName]))

    print("overNumStepEventNumber:", overNumStepEventNumber)
    print("lessNumEventNumber:", lessNumEventNumber)

    # TODO delete final zeros in timeOrderEventsArray and timeOrderLabelArray
    timeOrderEventsArray = timeOrderEventsArray[:timeOrderEventsArrayIndex,:,:]
    timeOrderLabelArray = timeOrderLabelArray[:timeOrderEventsArrayIndex]
    # print("timeOrderLabelArray", timeOrderLabelArray[-1])
    # print("timeOrderLabelArray", timeOrderLabelArray[-2])
    # print("timeOrderLabelArray", timeOrderLabelArray[0])
    print("falseNum", falseNum)
    print("trueNum", trueNum)




    return caseActivityDict, vocabulary, timeOrderEventsArray, timeOrderLabelArray


