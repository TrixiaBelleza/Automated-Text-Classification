from stackapi import StackAPI
import html2text
import pymysql

#database connection
def insert_into_train_db(id, question_body, python, javascript, java, c, r, while_loop, for_loop):
	connection = pymysql.connect(host="localhost",user="root",passwd="592008",database="questions_db" )
	cursor = connection.cursor()
	sql_insert_query = """ INSERT IGNORE INTO `Train_data`
					  (`id`, `question_body`, `python`, `javascript`, `java`, `c`, `r`, `while_loop`, `for_loop`) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
	insert_tuple = (id, question_body, python, javascript, java, c, r, while_loop, for_loop)
	result  = cursor.execute(sql_insert_query, insert_tuple)
	connection.commit()
	print ("Record inserted successfully into Train_data table")

#Main
SITE = StackAPI('stackoverflow', max_pages=11)
questions = SITE.fetch('questions', pages=6, tagged='r', filter='!)re8*vhaqGn7n9_0lKeP')

# print(questions["items"])
h = html2text.HTML2Text()
h.ignore_links = True

categories = ['python', 'javascript', 'java', 'c', 'r', 'while_loop', 'for_loop']
for i in range(len(questions["items"])):
	html_to_text = h.handle(questions["items"][i]["body"])
	question_body = html_to_text.lower()
	question_id = questions["items"][i]["question_id"]
	
	python = 0
	javascript = 0
	java = 0
	c = 0
	r = 0
	while_loop = 0
	for_loop = 0

	for j in range(len(questions["items"][i]["tags"])):
		if questions["items"][i]["tags"][j] == 'python':
			python = 1
		if questions["items"][i]["tags"][j] == 'javascript':
			javascript = 1
		if questions["items"][i]["tags"][j] == 'java':
			java = 1
		if questions["items"][i]["tags"][j] == 'c':
			c = 1
		if questions["items"][i]["tags"][j] == 'r':
			r = 1
		if questions["items"][i]["tags"][j] == 'while-loop':
			while_loop = 1
		if questions["items"][i]["tags"][j] == 'for-loop':
			for_loop = 1	
	insert_into_train_db(question_id, question_body, python, javascript, java, c, r, while_loop, for_loop)
