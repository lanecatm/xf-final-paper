INSERT INTO bpi2012_new([Case ID],[Activity],[Resource],[Complete Timestamp],[Variant],[Variant index],[Attribute 1],[concept:name],[lifecycle:transition]) select * from bpi2012;
'''
CREATE TABLE bpi2012(
    "Case ID" TEXT,
    "Activity" TEXT,
    "Resource" TEXT,
    "Complete Timestamp" TEXT,
    "Variant" TEXT,
    "Variant index" TEXT,
    "Attribute 1" TEXT,
    "concept:name" TEXT,
    "lifecycle:transition" TEXT
);
CREATE TABLE bpi2012_new(
    "ID" INTEGER PRIMARY KEY AUTOINCREMENT,
    "Case ID" TEXT,
    "Activity" TEXT,
    "Resource" TEXT,
    "Complete Timestamp" TEXT,
    "Variant" TEXT,
    "Variant index" TEXT,
    "Attribute 1" TEXT,
    "concept:name" TEXT,
    "lifecycle:transition" TEXT
);
'''

