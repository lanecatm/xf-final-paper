
def calculate_markov_array(dbName, tableName, activityColumnName, timeColumnName, caseColumnName, idColumnName):

    activity2IdDict = build_attribute_to_id_dict(activityColumnName, dbName, tableName):
    targetList = [caseColumnName, activityColumnName, timeColumnName]
    ansStr, attrName2IdDict, id2attrNameDict = attributeSqlStr(targetList)

    conn = sqlite3.connect(dbName)
    c = conn.cursor()
    print("calculate_markov_array connect db successfully")
    sql = "SELECT " + ansStr + " from " + tableName + " order by [" +  timeColumnName + "], [" + caseColumnName + "], [" + idColumnName + "]"
    print(sql)
    cursor = c.execute(sql)
    caseActivityDict = {}
    for row in cursor:
        caseName = row[attrName2IdDict[caseColumnName]]
        activity


    conn.close()
    activity_to_id = dict(zip(activityNameList, range(len(activityNameList))))
    print(activity_to_id)
    return activity_to_id



def calculate_simular_of_two_activity():

def calculate_all_array():

def find_first_(attributeName, dbName, tableName):
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

def load_data_from_db(activityColumnName, caseColumnName, timeColumnName, idColumnName, dbName, tableName, timeStrp):

    conn = sqlite3.connect(dbName)
    c = conn.cursor()
    print("connect db successfully")

    sql = "SELECT " + ansStr + " from " + tableName + " order by [" +  timeColumnName + "], [" + caseColumnName + "], [" + idColumnName + "]"
    print("sql:",sql)
    cursor = c.execute(sql)
    
    caseActivityDict = {}
    rowIndex = 1
    for row in cursor:
        if (rowIndex % 10000 == 0):
            print("input ", rowIndex)
            #print("input row:", row)
        rowIndex += 1
        caseName = row[attrName2IdDict[caseColumnName]]
        caseActivityDict[caseName].append(featureArray)
    conn.close()

    return caseActivityDict, vocabulary, timeOrderEventsArray, timeOrderLabelArray


