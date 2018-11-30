import time
import csv
import datetime
from business_calendar import Calendar, MO, TU, WE, TH, FR
import datetime
from predict import *
import json

# 0 Case ID
# 1 Activity
# 2 Resource
# 3 Complete Timestamp
# 4 Variant
# 5 Variant index
# 6 (case) IDofConceptCase
# 7 (case) Includes_subCases
# 8 (case) Responsible_actor
# 9 (case) SUMleges
# 10(case) caseProcedure
# 11(case) caseStatus
# 12(case) case_type
# 13(case) landRegisterID
# 14(case) last_phase
# 15(case) parts
# 16(case) requestComplete
# 17(case) termName
# 18action_code
# 19activityNameNL
# 20concept:name
# 21dateFinished
# 22dateStop
# 23dueDate
# 24lifecycle:transition
# 25monitoringResource
# 26planned
# 27question
import sqlite3


def convert_case_activity_list_from_db(dbName = "bpi2018.db", tableName = "BPIC2018_new",timeStrp = "%Y/%m/%d %H:%M:%S"):

    #caseAttributeNameList = ['(case) amount_applied0','(case) amount_applied1','(case) amount_applied2','(case) amount_applied3','(case) applicant','(case) application','(case) area','(case) basic payment','(case) cross_compliance','(case) department','(case) greening','(case) number_parcels','(case) payment_actual0','(case) payment_actual1','(case) payment_actual2','(case) payment_actual3','(case) penalty_ABP','(case) penalty_AGP','(case) penalty_AJLP','(case) penalty_AUVP','(case) penalty_AVBP','(case) penalty_AVGP','(case) penalty_AVJLP','(case) penalty_AVUVP','(case) penalty_B16','(case) penalty_B2','(case) penalty_B3','(case) penalty_B4','(case) penalty_B5','(case) penalty_B5F','(case) penalty_B6','(case) penalty_BGK','(case) penalty_BGKV','(case) penalty_BGP','(case) penalty_C16','(case) penalty_C4','(case) penalty_C9','(case) penalty_CC','(case) penalty_GP1','(case) penalty_JLP1','(case) penalty_JLP2','(case) penalty_JLP3','(case) penalty_JLP5','(case) penalty_JLP6','(case) penalty_JLP7','(case) penalty_V5','(case) penalty_amount0','(case) penalty_amount1','(case) penalty_amount2','(case) penalty_amount3','(case) program-id','(case) redistribution','(case) rejected','(case) risk_factor','(case) selected_manually','(case) selected_random','(case) selected_risk','(case) small farmer','(case) year','(case) young farmer']

    #activityAttributeList = ['doctype','note','subprocess','success', "concept:name"]
    conn = sqlite3.connect(dbName)
    c = conn.cursor()
    print("connect db successfully")
    cursor = c.execute("SELECT `id`, `case id`, `Activity`, `Complete Timestamp`  from " + tableName)
    caseActivityDict = {}
    caseActivityAppearTimesDict = {}
    activityCodeList = []
    rowIndex = 1
    for row in cursor:
        if (rowIndex % 200000 == 0):
            print("input ", rowIndex)
            break
        
        rowIndex += 1
        caseName = row[1]
        if not caseName in caseActivityDict:
            caseActivityDict[caseName] = []
         # 读取出每个activity出现次数
        if caseName in caseActivityAppearTimesDict:
            activityAppearTimesDict = caseActivityAppearTimesDict[caseName]
        else:
            activityAppearTimesDict = {}

        # 更新每个activity出现次数
        activityCode = row[2]
        if activityCode in activityAppearTimesDict:
            activityAppearTimesDict[activityCode] += 1
        else:
            activityAppearTimesDict[activityCode] = 1
        caseActivityAppearTimesDict[caseName] = activityAppearTimesDict

        # 不加上出现次数的activityCode
        #activityCodeList.append(activityCode)

        # 加上出现次数的activityCode
        activityCodeWithTime = activityCode + " & " + str(activityAppearTimesDict[activityCode])
        activityCodeList.append(activityCodeWithTime)

        #修改时间
        endTimeStr = row[3].split(".")[0]
        endTime = datetime.datetime.strptime(endTimeStr, timeStrp)

        activityInfo = {"id": row[0], "activity": row[2], "endTime": endTime, "activityCodeWithTime":activityCodeWithTime, "times":caseActivityAppearTimesDict[caseName][activityCode]}
        caseActivityDict[caseName].append(activityInfo)

    activityCodeSet = set(activityCodeList)
    activityCodeList = list(activityCodeSet)
    conn.close()
    return caseActivityDict

def find_activity_name_list(dbName = "bpi2018.db", tableName = "BPIC2018_new"):
    conn = sqlite3.connect(dbName)
    c = conn.cursor()
    print("connect db successfully")
    cursor = c.execute("SELECT distinct(`Activity`)  from " + tableName)
    activityNameList = []
    for row in cursor:
        activityNameList.append(row[0])
    conn.close()
    return activityNameList




def calcute_activity_times(dbName = "bpi2018.db", tableName = "BPIC2018_new"):
    activityNameList = find_activity_name_list(dbName, tableName)
    conn = sqlite3.connect(dbName)
    activityAppearCaseNumDict = {}
    for activityName in activityNameList:
        c = conn.cursor()
        sql = "SELECT count(distinct(`Case ID`))  from " + tableName + " where `Activity`='" + activityName + "'"
        #print(sql)
        cursor = c.execute(sql)
        for row in cursor:
            activityAppearCaseNumDict[activityName] = row[0]
            break
    for activityName, appearNum in activityAppearCaseNumDict.items():
        print(activityName, ":", appearNum)
    conn.close()

def calcute_activity_appear_times(dbName = "bpi2018.db", tableName = "BPIC2018_new"):
    activityNameList = find_activity_name_list(dbName, tableName)
    conn = sqlite3.connect(dbName)
    activityAppearAvgTimesDict = {}
    for activityName in activityNameList:
        c = conn.cursor()
        subSql = "SELECT count(*) as count_id from " + tableName + " where `Activity`='" + activityName + "' group by `case id`"
        sql = "SELECT AVG(count_id) from (" + subSql + ")"
        #sql = "SELECT count(distinct(`Case ID`))  from " + tableName + " where `Activity`='" + activityName + "'"
        #print(sql)
        cursor = c.execute(sql)
        for row in cursor:
            activityAppearAvgTimesDict[activityName] = row[0]
            break
    for activityName, appearNum in activityAppearAvgTimesDict.items():
        print(activityName, ":", appearNum)
    conn.close()



def calcute_two_activity_case_times(dbName = "bpi2018.db", tableName = "BPIC2018_new"):
    activityNameList = find_activity_name_list(dbName, tableName)
    conn = sqlite3.connect(dbName)
    activityAppearCaseNumDict = {}
    for activityName1 in activityNameList:
        subActivityAppearCaseNumDict = {}
        for activityName2 in activityNameList:
            c = conn.cursor()
            subsql = "SELECT distinct(`Case ID`)  from " + tableName + " where `Activity`='" + activityName1 + "'"
            sql = "SELECT COUNT(distinct(`Case ID`)) from " + tableName + " where `Activity`='" + activityName2 + "' and `Case ID` in (" + subsql + ")"
            #print(sql)
            cursor = c.execute(sql)
            for row in cursor:
                subActivityAppearCaseNumDict[activityName2] = row[0]
                break
            print(activityName1, "&", activityName2, ":", subActivityAppearCaseNumDict[activityName2])
        activityAppearCaseNumDict[activityName1] = subActivityAppearCaseNumDict

    #for activityName1, subActivityAppearCaseNumDict in activityAppearCaseNumDict.items():
    #    for activityName2, appearNum in subActivityAppearCaseNumDict.items():
    #        print(activityName1, "&", activityName2, ":", appearNum)
    conn.close()
    return activityAppearCaseNumDict



#activityAppearCaseNumDict = calcute_two_activity_case_times()
'''
tJson = json.dumps(activityAppearCaseNumDict)
with open('./activity_num_all.txt', 'w') as f:
    f.write(tJson)

exit()
'''
with open('./activity_num.txt', 'r') as f:
    tJson = f.read()
    activityAppearCaseNumDict = json.loads(tJson)
activityNameList = sorted(activityAppearCaseNumDict)
for activityName in activityNameList:
    print(activityName)
activityRootList = list(range(len(activityNameList)))

for i in range(len(activityNameList)):
    for j in range(i + 1, len(activityNameList)):
        activityName1 = activityNameList[i]
        activityName2 = activityNameList[j]
        if( activityAppearCaseNumDict[activityName1][activityName2] / float(activityAppearCaseNumDict[activityName1][activityName1]) > 0.9 and 
            activityAppearCaseNumDict[activityName1][activityName2] / float(activityAppearCaseNumDict[activityName2][activityName2]) > 0.9):
            activityNameIndex1 = activityNameList.index(activityName1)
            activityNameIndex2 = activityNameList.index(activityName2)
            root1 = activityNameIndex1
            while activityRootList[root1] != root1:
                root1 = activityRootList[root1]
            root2 = activityNameIndex2
            while activityRootList[root2] != root2:
                root2 = activityRootList[root2]
            activityRootList[root2] = root1
        else:
            activityNameIndex1 = activityNameList.index(activityName1)
            activityNameIndex2 = activityNameList.index(activityName2)
            root1 = activityNameIndex1
            while activityRootList[root1] != root1:
                root1 = activityRootList[root1]
            root2 = activityNameIndex2
            while activityRootList[root2] != root2:
                root2 = activityRootList[root2]
            if(root1 == root2):
                print("error ", activityName1 , " & " , activityName2, ":", activityAppearCaseNumDict[activityName1][activityName2], " ", activityAppearCaseNumDict[activityName1][activityName1], " ", activityAppearCaseNumDict[activityName2][activityName2])

sameAppearListDict = {}
for i in range(len(activityNameList)):
    if activityRootList[i] == i:
        sameAppearListDict[activityNameList[i]] = [activityNameList[i]]
    else:
        root1 = i
        while activityRootList[root1] != root1:
            root1 = activityRootList[root1]
        sameAppearListDict[activityNameList[root1]].append(activityNameList[i])

for activityNameRoot, activitySameAppearList in sameAppearListDict.items():
    if activityAppearCaseNumDict[activityNameRoot][activityNameRoot] < 100:
        continue
    print("------------------------")
    for activityName in activitySameAppearList:
        print(activityName, ":", activityAppearCaseNumDict[activityName][activityName])












exit()

caseActivityDict = convert_case_activity_list_from_db()
activityNameList = find_activity_name_list()

timeStampAndIdDict = {}
caseActivitySequenceDict = {}
caseActivityTimeDict = {}
for caseName, activityInfoList in caseActivityDict.items():
    if (len(activityInfoList) > 200):
        continue
    if not caseName in caseActivitySequenceDict:
        caseActivitySequenceDict[caseName] = []
        caseActivityTimeDict[caseName] = []
    for activityAppearIndex, activityInfo in enumerate(activityInfoList):
        if not len(caseActivityTimeDict[caseName]) == 0:
            if caseActivityTimeDict[caseName][-1] > activityInfo["endTime"]:
                print("error sequence, ", caseName, " ", activityInfo)
                print(activityInfoList)
        caseActivityTimeDict[caseName].append(activityInfo["endTime"])
        activityNameIndex = activityNameList.index(activityInfo["activity"])
        caseActivitySequenceDict[caseName].append(activityNameIndex)
    
        activityTimestamp = int(time.mktime(activityInfo["endTime"].timetuple()))
        timeStampAndIdDict[str(activityTimestamp) + "_" + caseName + "_" + activityInfo["activity"]] = {"caseName": caseName, "activityAppearIndex": activityAppearIndex, "activity":activityInfo["activity"]}
    

caseActivityStartTimeDict = {}
caseActivityEndTimeDict = {}
cal = Calendar()
for caseName, activityTimeList in caseActivityTimeDict.items():
    startTime = activityTimeList[0]
    endTime = activityTimeList[-1]
    caseActivityStartTimeDict[caseName] = []
    caseActivityEndTimeDict[caseName] = []
    for activityTime in activityTimeList:
        startWorkday = cal.workdaycount(startTime.date(), activityTime.date())
        endWorkday = cal.workdaycount(activityTime.date(), endTime.date())
        caseActivityStartTimeDict[caseName].append(startWorkday)
        caseActivityEndTimeDict[caseName].append(endWorkday)
#print(caseActivityStartTimeDict["d0c4a4241daa1d89"])
#print(caseActivityEndTimeDict["d0c4a4241daa1d89"])


sortedTimeStampAndIdList = sorted(timeStampAndIdDict)
#print(sortedTimeStampAndIdList[0:10])
#print(sortedTimeStampAndIdList[-10:])
subSequenceList = []
subSequenceTimeList = []
for timeStampKey in sortedTimeStampAndIdList:
    timeStampAndIdInfo = timeStampAndIdDict[timeStampKey]
    caseName = timeStampAndIdInfo["caseName"]
    activityAppearIndex = timeStampAndIdInfo["activityAppearIndex"]
    tmpList = [-1] * 200
    tmpList[0:activityAppearIndex + 1] = caseActivitySequenceDict[caseName][0:activityAppearIndex + 1] 

    #if activityAppearIndex > 10:
    #    print(tmpList)
    subSequenceList.append(tmpList)
    if activityAppearIndex == len(caseActivitySequenceDict[caseName]) -1:
        subSequenceTimeList.append(0)
    else:
        subSequenceTimeList.append(caseActivityStartTimeDict[caseName][activityAppearIndex + 1] - caseActivityStartTimeDict[caseName][activityAppearIndex])
    
import numpy as np
subSequenceArr = np.array(subSequenceList) + 1
print(subSequenceArr[0:10])
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
keyList = []
np.set_printoptions(threshold=-1)
for i in range(0, 200):
    keyList.append([])
    for j in range(0, 200):
        keyList[i].append((i+j)%171)
enc.fit(keyList)
subSequenceOneHotArray = enc.transform(subSequenceArr).toarray()

KNN_step_by_step(subSequenceArr, np.array(subSequenceTimeList), 0.6, 4)







#caseActivityDict = convert_case_activity_list_from_db()
#print("all case:", len(caseActivityDict))
#print(caseActivityDict["8b99873a6136cfa6"])

