import pymysql

#database connection
connection = pymysql.connect(host="localhost",user="root",passwd="592008",database="questions_db" )
cursor = connection.cursor()

TrainDataTbl = """CREATE TABLE Train_data(
id VARCHAR(500) PRIMARY KEY,
question_body  TEXT(65000) NOT NULL,
python INT(50),
javascript INT(50),
java INT(50),
c INT(50),
r INT(50),
while_loop INT(50),
for_loop INT(50))"""

cursor.execute(TrainDataTbl)
connection.close()