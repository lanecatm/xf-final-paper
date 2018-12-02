CREATE Table bpi2014_new(
    "ID" INTEGER PRIMARY KEY AUTOINCREMENT,
    "Incident ID" TEXT,
    "DateStamp" TEXT,
    "IncidentActivity_Number" TEXT,
    "IncidentActivity_Type" TEXT,
    "Assignment Group" TEXT,
    "KM number" TEXT,
    "Interaction ID" TEXT, 
    "DateTime_New" DATETIME
);

INSERT INTO bpi2014_new([Incident ID],[DateStamp],[IncidentActivity_Number],[IncidentActivity_Type],[Assignment Group],[KM number],[Interaction ID],[DateTime_New]) select * from bpi2014;
'''
CREATE Table bpi2014(
    "Incident ID" TEXT,
    "DateStamp" TEXT,
    "IncidentActivity_Number" TEXT,
    "IncidentActivity_Type" TEXT,
    "Assignment Group" TEXT,
    "KM number" TEXT,
    "Interaction ID" TEXT, 
    "DateTime_New" DATETIME
);
'''

