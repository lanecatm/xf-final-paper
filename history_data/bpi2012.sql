CREATE TABLE bpi2012(
    "Case ID" TEXT,
    "Activity" TEXT,
    "Resource" TEXT,
    "Complete Timestamp" TEXT,
    "Variant" TEXT,
    "Variant index" TEXT,
    "(case) AMOUNT_REQ" TEXT,
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
    "(case) AMOUNT_REQ" TEXT,
    "concept:name" TEXT,
    "lifecycle:transition" TEXT
);

.mode csv
.import bpi2012_1.csv bpi2012
INSERT INTO bpi2012_new([Case ID],[Activity],[Resource],[Complete Timestamp],[Variant],[Variant index],[(case) AMOUNT_REQ],[concept:name],[lifecycle:transition]) select * from bpi2012;
drop table bpi2012;

