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

	# plt.savefig('./scores/' + score_type + '_perfold2.png')

def ML_algorithms(x_train, x_test, train_category, test_category, category, algorithm_type):
	
	filename = '2' + algorithm_type + '-' + category + '.sav'
	loaded_model = pickle.load(open('./data-models/' + filename, 'rb'))

	# compute the testing accuracy 
	prediction = loaded_model.predict(x_test)

	print(algorithm_type + " Prediction:")
	# print(prediction)
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

# Connect to the database
db_connection = pymysql.connect(host='localhost',
							 user='root',
							 password='592008',
							 db='questions_db',
							 charset='utf8mb4',
							 cursorclass=pymysql.cursors.DictCursor)
categories = ['python', 'javascript', 'java', 'c', 'r', 'mysql', 'html', 'if_statement', 'while_loop', 'for_loop', 'css']
df = pd.read_sql('SELECT * FROM test_data', con=db_connection)

db_connection.close()

vectorizer = pickle.load(open('./data-models/vectorizer.sav', 'rb'))

kf = KFold(n_splits=10)
ave_f1scores = {}
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
	
	train = df.iloc[train_index]
	test = df.iloc[test_index]
	train_text = train['question_body']
	test_text = test['question_body']
	
	# vectorizer.fit(train_text)

	x_train = vectorizer.transform(train_text)
	y_train = train.drop(labels = ['question_body'], axis=1)

	x_test = vectorizer.transform(test_text)
	y_test = test.drop(labels = ['question_body'], axis=1)

	for category in categories:
		print('... Processing {}'.format(category))
		# train the SVC model using X_dtm & y
		f1score, accuracy, recall, precision = ML_algorithms(x_train, x_test, train[category], test[category], category, 'svc')
		fold_svcf1scores += f1score
		fold_svcaccuracy += accuracy
		fold_svcrecall += recall
		fold_svcprecision += precision

		key_name = "svc_" + category
		if key_name in ave_f1scores:
			ave_f1scores[key_name] += f1score
		else :
			ave_f1scores[key_name] = f1score

		# train the LogReg model using X_dtm & y
		f1score, accuracy, recall, precision = ML_algorithms(x_train, x_test, train[category], test[category], category, 'lr')
		fold_logregf1scores += f1score
		fold_logregaccuracy += accuracy
		fold_logregrecall += recall
		fold_logregprecision += precision

		key_name = "logreg_" + category
		if key_name in ave_f1scores:
			ave_f1scores[key_name] += f1score
		else :
			ave_f1scores[key_name] = f1score

		# train the NB model using X_dtm & y
		f1score, accuracy, recall, precision = ML_algorithms(x_train, x_test, train[category], test[category], category, 'nb')
		fold_nbf1scores += f1score
		fold_nbaccuracy += accuracy
		fold_nbrecall += recall
		fold_nbprecision += precision
	
		key_name = "nb_" + category
		if key_name in ave_f1scores:
			ave_f1scores[key_name] += f1score
		else :
			ave_f1scores[key_name] = f1score

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

ave_f1scores = {k: v / 10 for k, v in ave_f1scores.items()} #divide each value in dictionary by 10, to get the average score out of all 10 folds.
svc_fscores = ()
logreg_fscores = ()
nb_fscores = ()

for category in categories:
	svc_fscores += (ave_f1scores.get('svc_'+category),)
	logreg_fscores += (ave_f1scores.get('logreg_'+category),)
	nb_fscores += (ave_f1scores.get('nb_'+category),)
	print(category)

n_groups = 11
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.3
opacity = 0.8

rects1 = plt.bar(index, svc_fscores, bar_width,
alpha=opacity,
color='b',
label='SVM')

rects2 = plt.bar(index + bar_width, logreg_fscores, bar_width,
alpha=opacity,
color='g',
label='LR')

rects3 = plt.bar(index + bar_width + bar_width, nb_fscores, bar_width,
alpha=opacity,
color='y',
label='NB')

categories_tuple = ('Python', 'JS', 'Java', 'C', 'R', 'MySQL', 'HTML', 'If', 'While', 'For', 'CSS')
plt.xlabel('Tags')
plt.ylabel('F Scores')
plt.title('Distribution of F Scores for Each Tag')
plt.xticks(index + bar_width, categories_tuple)
plt.legend()
 
plt.tight_layout()
# plt.savefig('./scores/avefscores3.png')