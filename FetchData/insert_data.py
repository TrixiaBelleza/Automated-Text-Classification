from stackapi import StackAPI
import pymysql
import pickle
import datapreprocessing 

def insert_into_db(id, question_body, categories_dict) :
	connection = pymysql.connect(host="localhost",user="root",passwd="592008",database="questions_db" )
	cursor = connection.cursor()	

	#Generate SQL Insert Query
	sql_insert_query = " REPLACE INTO `training_data` (`id`, `question_body`, "
	format_query = "(%s, %s,"
	insert_tuple = (id, question_body)
	for key in categories_dict:
		sql_insert_query += "`" + key + "`,"
		format_query += "%s,"
		insert_tuple += (categories_dict[key], )
	sql_insert_query = sql_insert_query[:-1] #remove extra comma		
	format_query = format_query[:-1] #remove extra comma
	sql_insert_query += ") VALUES " + format_query + ")"


	final_sql_insert_query = " "" " + sql_insert_query + " "" "
	result = cursor.execute(sql_insert_query, insert_tuple)
	connection.commit()

	print("Successfully inserted to new_data")
	connection.close()

def add_col_to_db(col_name):
	connection = pymysql.connect(host="localhost",user="root",passwd="592008",database="questions_db" )
	cursor = connection.cursor()

	sql_query = "ALTER TABLE `new_data` ADD %s INT(50) DEFAULT 0" % (col_name)
	cursor.execute(sql_query)
	connection.close()

def tag_map(questions, categories_dict):
	for i in range(len(questions["items"])):
		html_to_text = datapreprocessing.h.handle(questions["items"][i]["body"])
		question_body = datapreprocessing.cleanPunc(html_to_text.lower())		#Remove punctuation marks
		question_body = datapreprocessing.stemming(question_body)				#Get root words
		question_id = questions["items"][i]["question_id"]

		for j in range(len(questions["items"][i]["tags"])):
			key = questions["items"][i]["tags"][j]
			if key in categories_dict:
				print("categories_dict[key]: ")
				print(categories_dict[key])
				categories_dict[key] = 1
		insert_into_db(question_id, question_body, categories_dict)

def scrape_all(categories_dict):
	for key in categories_dict:
		scrape(key, categories_dict)
		
def scrape(tag, categories_dict):
	if '_' in tag:
		tag = tag.replace('_', '-')

	SITE = StackAPI('stackoverflow', max_pages=3)
	questions = SITE.fetch('questions', page=2, tagged=tag, filter='!)re8*vhaqGn7n9_0lKeP')

	tag_map(questions, categories_dict)

def add_new_tag(categories_dict):
	new_tag = input("Enter new tag: ")
	if '-' in new_tag:
		new_tag = new_tag.replace('-', '_') #change dash to underscore for db to accept the column name
	categories_dict.update({new_tag : 0}) #Append to categories_dict
	add_col_to_db(new_tag) #Add new column to table

	scrape(new_tag, categories_dict)
	return categories_dict
def menu():
	print("[1] Add New Tag")
	print("[2] Scrape for all tags")
	print("[3] Exit")
	choice = int(input("Enter choice: "))
	return choice

choice = menu()
#Load categories from file
pickle_in = open("categories_dict.pickle","rb")
categories_dict = pickle.load(pickle_in)
while choice != 3:
	if choice == 1:
		categories_dict = add_new_tag(categories_dict)
	if choice == 2:
		scrape_all(categories_dict)
	if choice > 3 or choice < 1:
		print("Please choose from the menu.")	
	choice = menu()

categories_dict.fromkeys(categories_dict, 0) #Reset all to 0 
#Save all categories to categories file 
pickle_out = open("categories_dict.pickle","wb")
pickle.dump(categories_dict, pickle_out)
pickle_out.close()

