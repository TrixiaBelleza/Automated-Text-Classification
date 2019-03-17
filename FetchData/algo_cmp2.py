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

def create_10fold_df(fold_svc_score_list, fold_logreg_score_list, fold_nb_score_list, score_type):
    score_df_results = np.array([['FOLDS','SVM','LR', 'NB'],
                ['Fold1', fold_svc_score_list[0], fold_logreg_score_list[0], fold_nb_score_list[0]],
                ['Fold2', fold_svc_score_list[1], fold_logreg_score_list[1], fold_nb_score_list[1]],
                ['Fold3', fold_svc_score_list[2], fold_logreg_score_list[2], fold_nb_score_list[2]],
                ['Fold4', fold_svc_score_list[3], fold_logreg_score_list[3], fold_nb_score_list[3]],
                ['Fold5', fold_svc_score_list[4], fold_logreg_score_list[4], fold_nb_score_list[4]],
                ['Fold6', fold_svc_score_list[5], fold_logreg_score_list[5], fold_nb_score_list[5]],
                ['Fold7', fold_svc_score_list[6], fold_logreg_score_list[6], fold_nb_score_list[6]],
                ['Fold8', fold_svc_score_list[7], fold_logreg_score_list[7], fold_nb_score_list[7]],
                ['Fold9', fold_svc_score_list[8], fold_logreg_score_list[8], fold_nb_score_list[8]],
                ['Fold10', fold_svc_score_list[9], fold_logreg_score_list[9], fold_nb_score_list[9]],
          	    ['Average', round(sum(fold_svc_score_list)/10, 5), round(sum(fold_logreg_score_list)/10, 5), round(sum(fold_nb_score_list)/10, 5)]
                ])

    score_df = pd.DataFrame(data=score_df_results[1:,1:],
                      index=score_df_results[1:,0],
                      columns=score_df_results[0,1:])
    fig, ax = plt.subplots(figsize=(12, 2)) # set size frame
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis
    ax.axis('off')
    table(ax, score_df, loc='center', colWidths=[0.17]*len(score_df.columns))  # where df is your data frame

    plt.savefig('./scores/' + score_type + '_perfold.png')

def ML_algorithms(x_train, x_test, train_category, test_category, algorithm_type):
	if algorithm_type == 'SVC':
		# Using pipeline for applying linearSVC and one vs rest classifier
		pipeline = Pipeline([
						('clf', OneVsRestClassifier(LinearSVC(), n_jobs=-1)),
					])
	if algorithm_type == 'LR':
		# Using pipeline for applying logistic regression and one vs rest classifier
		pipeline = Pipeline([
						('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),
					])
	if algorithm_type == 'NB':
		# Using pipeline for applying Gaussian Naive Bayes and one vs rest classifier
		pipeline = Pipeline([
						('clf', OneVsRestClassifier(MultinomialNB(alpha=5), n_jobs=-1)),
					])
	pipeline.fit(x_train, train_category)
	
	# compute the testing accuracy 
	prediction = pipeline.predict(x_test)
	print(algorithm_type + " Prediction:")
	print(prediction)
	print('Test F-SCORE is {}'.format(f1_score(test_category, prediction, average='macro')))
	f1score = f1_score(test_category, prediction, average='macro')
	print("\n")
	print('Test ACCURACY is {}'.format(accuracy_score(test_category, prediction)))
	accuracy = accuracy_score(test_category, prediction)
	print("\n")
	print('Test RECALL is {}'.format(recall_score(test_category, prediction, average='macro')))
	recall = recall_score(test_category, prediction, average='macro')
	print("\n")
	print('Test PRECISION is {}'.format(precision_score(test_category, prediction, average='macro')))
	precision = precision_score(test_category, prediction, average='macro')
		
	return f1score, accuracy, recall, precision

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

kf = KFold(n_splits=10)
fold_svcf1scores_list = []
fold_svcaccuracy_list = []
fold_svcrecall_list = []
fold_svcprecision_list = []

fold_logregf1scores_list = []
fold_logregaccuracy_list = []
fold_logregrecall_list = []
fold_logregprecision_list = []

fold_nbf1scores_list = []
fold_nbaccuracy_list = []
fold_nbrecall_list = []
fold_nbprecision_list = []
for train_index, test_index in kf.split(df):

	fold_svcaccuracy = 0
	fold_svcf1scores = 0
	fold_svcrecall = 0
	fold_svcprecision = 0

	fold_logregaccuracy = 0
	fold_logregf1scores = 0
	fold_logregrecall = 0
	fold_logregprecision = 0

	fold_nbaccuracy = 0
	fold_nbf1scores = 0
	fold_nbrecall = 0
	fold_nbprecision = 0
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

	for category in categories:
		print('... Processing {}'.format(category))
		# train the SVC model using X_dtm & y
		f1score, accuracy, recall, precision = ML_algorithms(x_train, x_test, train[category], test[category], 'SVC')
		fold_svcf1scores += f1score
		fold_svcaccuracy += accuracy
		fold_svcrecall += recall
		fold_svcprecision += precision

		# train the LogReg model using X_dtm & y
		f1score, accuracy, recall, precision = ML_algorithms(x_train, x_test, train[category], test[category], 'LR')
		fold_logregf1scores += f1score
		fold_logregaccuracy += accuracy
		fold_logregrecall += recall
		fold_logregprecision += precision

		# train the SVC model using X_dtm & y
		f1score, accuracy, recall, precision = ML_algorithms(x_train, x_test, train[category], test[category], 'NB')
		fold_nbf1scores += f1score
		fold_nbaccuracy += accuracy
		fold_nbrecall += recall
		fold_nbprecision += precision
		
		
	fold_svcf1scores_list.append(round((fold_svcf1scores/len(categories))*100, 5))
	fold_svcaccuracy_list.append(round((fold_svcaccuracy/len(categories))*100, 5))
	fold_svcrecall_list.append(round((fold_svcrecall/len(categories))*100, 5))
	fold_svcprecision_list.append(round((fold_svcprecision/len(categories))*100, 5))

	fold_logregf1scores_list.append(round((fold_logregf1scores/len(categories))*100, 5))
	fold_logregaccuracy_list.append(round((fold_logregaccuracy/len(categories))*100, 5))
	fold_logregrecall_list.append(round((fold_logregrecall/len(categories))*100, 5))
	fold_logregprecision_list.append(round((fold_logregprecision/len(categories))*100, 5))

	fold_nbf1scores_list.append(round((fold_nbf1scores/len(categories))*100, 5))
	fold_nbaccuracy_list.append(round((fold_nbaccuracy/len(categories))*100, 5))
	fold_nbrecall_list.append(round((fold_nbrecall/len(categories))*100, 5))
	fold_nbprecision_list.append(round((fold_nbprecision/len(categories))*100, 5))

create_10fold_df(fold_svcf1scores_list, fold_logregf1scores_list, fold_nbf1scores_list, 'fscore')
create_10fold_df(fold_svcaccuracy_list, fold_logregaccuracy_list, fold_nbaccuracy_list, 'acc')
create_10fold_df(fold_svcrecall_list, fold_logregrecall_list, fold_nbrecall_list, 'recall')
create_10fold_df(fold_svcprecision_list, fold_logregprecision_list, fold_nbprecision_list, 'precision')
