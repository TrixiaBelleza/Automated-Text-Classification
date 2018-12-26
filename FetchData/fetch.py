import pymysql

#database connection
connection = pymysql.connect(host="localhost",user="root",passwd="592008",database="questions_db" )
cursor = connection.cursor()

QuestionsTbl = """CREATE TABLE Questions(
ID VARCHAR(500) PRIMARY KEY,
question_body  TEXT(65000) NOT NULL,
tags VARCHAR(500))"""


insert1 = "INSERT INTO Questions(ID, question_body, TAGS) VALUES ('1', 'hello?', 'python,machine learning');"
# cursor.execute(QuestionsTbl)
# cursor.execute(insert1)


# queries for retrievint all rows
retrive = "Select * from Questions;"

#executing the quires
cursor.execute(retrive)
rows = cursor.fetchall()
for row in rows:
   print(row)


connection.commit()
connection.close()