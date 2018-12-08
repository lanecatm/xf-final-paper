import json
import os


def save_data_from_db(dataList, dirPath = "./", description = "", version = "0"):
    filePath = dirPath + description + "_V" + version + ".txt"
    tJson = json.dumps(dataList)
    with open(filePath, 'w') as f:
        f.write(tJson)
    return

def load_same_data_from_txt(dirPath = "./", description = "", version = "0"):
    filePath = dirPath + description + "_V" + version + ".txt"
    with open(filePath, 'r') as f:
        tJson = f.read()
        dataList = json.loads(tJson)
        print("load:", dataList)
        #return dataList[0], dataList[1], dataList[2]
        return dataList



