import pymysql
import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Connect to the database
db_connection = pymysql.Connect(host='localhost',
							 user='root',
							 password='592008',
							 db='questions_db',
							 charset='utf8mb4',
							 cursorclass=pymysql.cursors.DictCursor)
# categories = ['python/', 'javascript/', 'java/', 'c/', 'r/', 'mysql/', 'html/', 'if_statement/', 'while_loop', 'for_loop', 'css']
pickle_in = open("categories_dict.pickle","rb")
categories = pickle.load(pickle_in)

df = pd.read_sql('SELECT * FROM test_data', con=db_connection)
db_connection.close()

test_text = df['question_body']
# print(test_text)

vectorizer = pickle.load(open('../extension-app/models/vectorizer.sav', 'rb'))

x_test = vectorizer.transform(test_text)
y_test = df.drop(labels = ['question_body'], axis=1)

for category in categories:
	print('... Processing {}'.format(category))
	filename = 'svc-' + category + '.sav'
	loaded_model = pickle.load(open('../extension-app/models/' + filename, 'rb'))

	prediction = loaded_model.predict(x_test)

	print("SVC Prediction:")
	print(prediction)
	print('Test weighted F-SCORE is {}'.format(f1_score(df[category], prediction, average='weighted')))
	print('Test micro F-SCORE is {}'.format(f1_score(df[category], prediction, average='micro')))
	print('Test macro F-SCORE is {}'.format(f1_score(df[category], prediction, average='macro')))

