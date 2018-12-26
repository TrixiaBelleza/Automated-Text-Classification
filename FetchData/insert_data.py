from stackapi import StackAPI
import html2text
import pymysql

#database connection
def insert_into_questions_db(question_id, question_body, tags):
	connection = pymysql.connect(host="localhost",user="root",passwd="592008",database="questions_db" )
	cursor = connection.cursor()
	sql_insert_query = """ INSERT IGNORE INTO `Questions`
					  (`id`, `question_body`, `tags`) VALUES (%s,%s,%s)"""
	insert_tuple = (question_id, question_body, tags)
	result  = cursor.execute(sql_insert_query, insert_tuple)
	connection.commit()
	print ("Record inserted successfully into Questions table")

#Main
SITE = StackAPI('stackoverflow', max_pages=100)
questions = SITE.fetch('questions', tagged='javascript', filter='!)re8*vhaqGn7n9_0lKeP')

# print(questions["items"])
h = html2text.HTML2Text()
h.ignore_links = True

# print((questions["items"][1]["tags"]))
for i in range(len(questions["items"])):
	html_to_text = h.handle(questions["items"][i]["body"])
	question_body = html_to_text.lower()
	question_id = questions["items"][i]["question_id"]
	tags = ""
	for j in range(len(questions["items"][i]["tags"])):
		if j != 0:
			tags = tags + "," + questions["items"][i]["tags"][j]
		else :
			tags = questions["items"][i]["tags"][j]
	insert_into_questions_db(question_id, question_body, tags)
