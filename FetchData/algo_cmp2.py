import pymysql
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table
import numpy as np
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
from sklearn.model_selection import KFold


stop_words =  nltk.corpus.stopwords.words('english')
new_stop_words = ['(', ')', '[', ']', '{', '}', '"', "'", '``', '""',"''", ',', '.', '“', '”', '’', '`']
stop_words.extend(new_stop_words)

# Connect to the database
db_connection = pymysql.connect(host='localhost',
							 user='root',
							 password='592008',
							 db='questions_db',
							 charset='utf8mb4',
							 cursorclass=pymysql.cursors.DictCursor)
categories = ['python', 'javascript', 'java', 'c', 'r', 'mysql', 'html', 'if_statement', 'while_loop', 'for_loop', 'css']
df = pd.read_sql('SELECT * FROM complete_train_data2', con=db_connection)

db_connection.close()

kf = KFold(n_splits=2)
fold_svcf1scores_list = []

for train_index, test_index in kf.split(df):


	fold_svcf1scores = 0
	print("TRAIN:", train_index, "TEST:", test_index)

	train = df.iloc[train_index]
	test = df.iloc[test_index]
	train_text = train['question_body']
	test_text = test['question_body']
	
	vectorizer = TfidfVectorizer(strip_accents='unicode', stop_words=stop_words, analyzer='word', ngram_range=(1,3), norm='l2')
	vectorizer.fit(train_text)

	x_train = vectorizer.transform(train_text)
	y_train = train.drop(labels = ['question_body'], axis=1)

	x_test = vectorizer.transform(test_text)
	y_test = test.drop(labels = ['question_body'], axis=1)

	# Using pipeline for applying linearSVC and one vs rest classifier
	SVC_pipeline = Pipeline([
					('clf', OneVsRestClassifier(LinearSVC(), n_jobs=-1)),
				])
	# Using pipeline for applying logistic regression and one vs rest classifier
	LogReg_pipeline = Pipeline([
					('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),
				])
	# Using pipeline for applying Gaussian Naive Bayes and one vs rest classifier
	NB_pipeline = Pipeline([
					('clf', OneVsRestClassifier(MultinomialNB(alpha=5), n_jobs=-1)),
				])

	for category in categories:
		print('... Processing {}'.format(category))
		 # train the SVC model using X_dtm & y
		SVC_pipeline.fit(x_train, train[category])

		# compute the testing accuracy of SVC
		svc_prediction = SVC_pipeline.predict(x_test)
		print("SVC Prediction:")
		print(svc_prediction)
		print('Test F-SCORE is {}'.format(f1_score(test[category], svc_prediction, average='macro')))
		fold_svcf1scores += f1_score(test[category], svc_prediction, average='macro')
		print("\n")

	print("Total fold_svcf1scores: ")
	print(fold_svcf1scores/len(categories))