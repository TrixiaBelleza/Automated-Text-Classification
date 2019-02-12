from stackapi import StackAPI
import html2text
import pymysql
import re
import pickle
from nltk.stem.snowball import SnowballStemmer

#database connection
def insert_into_train_db(id, question_body, python, javascript, java, c, r, while_loop, for_loop):
	connection = pymysql.connect(host="localhost",user="root",passwd="592008",database="questions_db" )
	cursor = connection.cursor()
	sql_insert_query = """ INSERT IGNORE INTO `clean_train_data`
					  (`id`, `question_body`, `python`, `javascript`, `java`, `c`, `r`, `while_loop`, `for_loop`) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
	insert_tuple = (id, question_body, python, javascript, java, c, r, while_loop, for_loop)
	result  = cursor.execute(sql_insert_query, insert_tuple)
	connection.commit()
	print ("Record inserted successfully into create_train_data table")

#Main
SITE = StackAPI('stackoverflow', max_pages=30)
questions = SITE.fetch('questions', page=20, tagged='while-loop', filter='!)re8*vhaqGn7n9_0lKeP')

############ DATA PREPROCESSING ################
stemmer = SnowballStemmer("english")

def cleanPunc(sentence): #function to clean the word of unnecessary punctuation or special characters using re library or regex
    cleaned = re.sub(r'[?|!|,|~|^]',r'',sentence)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence
# print(questions["items"])
h = html2text.HTML2Text()
h.ignore_links = True

categories = ['python', 'javascript', 'java', 'c', 'r', 'while_loop', 'for_loop']
for i in range(len(questions["items"])):
	html_to_text = h.handle(questions["items"][i]["body"])
	question_body = cleanPunc(html_to_text.lower())		#Remove punctuation marks
	question_body = stemming(question_body)				#Get root words
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
