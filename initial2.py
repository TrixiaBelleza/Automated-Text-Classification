import json
import html2text
import math
import nltk
import re #regEx
import pandas as pd
from scipy import sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import numpy as np
nltk.download('stopwords')

stop_words =  nltk.corpus.stopwords.words('english')
new_stop_words = ['(', ')', '[', ']', '{', '}', '"', "'", '``', '""',"''", ',', '.', '“', '”', '’', '`']
stop_words.extend(new_stop_words)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

#read the data.json file which contains the tags and the questions in stack overflow
with open("data.json", "r") as read_file:
    data = json.load(read_file)

h = html2text.HTML2Text()
h.ignore_links = True

'''
training_data = [
				{'question_body' : blabla question1?, 'tags':[python, if-statement]},
				{{'question_body' : blabla question1?, 'tags':[java, for-loop]},
				{'question_body' : blabla question1?, 'tags':[java, if-statement]
				]
'''
training_data = []
categories = ['python', 'if-statement', 'for-loop', 'java']

for i in range(len(data["items"])):
	body_tag_dict = {'question_body' : '', 'tags' : []}
	#Convert html code to text since data["items"][i]["body"] returns something like this: "<p>I have 2 columns <p>""
	html_to_text = h.handle(data["items"][i]["body"])
	html_to_text = html_to_text.lower()
	clean_text = preprocess_text(html_to_text)

	body_tag_dict['question_body'] = clean_text
	for j in range(len(data["items"][i]["tags"])):
		body_tag_dict['tags'].append(data["items"][i]["tags"][j])

	training_data.append(body_tag_dict)

question_body_array = []
tags_array = []
for i in range(len(training_data)) :
   question_body_array.append(training_data[i].get("question_body"))
   tags_array.append(training_data[i].get("tags"))
# print(question_body_array)
print(tags_array)
x_df = pd.DataFrame(question_body_array, columns={'body'})

print(x_df)


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(x_df.body)
print(X)

y_df = pd.DataFrame(tags_array)
y_df['tags'] = y_df.apply(lambda r: tuple(r), axis=1).apply(np.array)
y_df = y_df['tags'].tolist()
print(y_df)

LSVC = LinearSVC()
mlb = OneVsRestClassifier(LSVC, n_jobs=None)

scores = cross_val_score(mlb, X, y_df, cv=KFold(n_splits=10, shuffle=True), scoring="accuracy")
print("Mean Score for %s: %.4f" %("accuracy", scores.mean()))
# df = pd.DataFrame(training_data[0,:])

# print(df)
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(df.question_body)
# X = sp.csr_matrix(X)
# X = X.todense()
# print(X)
# df.question_body = X
# print(df)
# # df.question_body = df.question_body.todense()

# #train and test data split from scikit
# train, test = train_test_split(df, test_size=0.33, shuffle=True)
# print(train)
# x_train = train.question_body
# y_train = train.tags


# LSVC = LinearSVC()
# mlb = OneVsRestClassifier(LSVC, n_jobs=None)
# y = MultiLabelBinarizer().fit_transform(y_train)
# # # # print(X)
# # # # X = sp.csr_matrix(X)

# # # # print(y_train.shape)
# # # # print(X)

# mlb.fit(x_train,y)
# prediction = mlb.predict(X_test)
# print('Test accuracy is:')

# # for i in range(len(categories)):
# #     print('... Processing ' + categories[i])
#     # train the model using X_dtm & y
#     # compute the testing accuracy
#     # prediction = NB_pipeline.predict(X_test)
#     # print('Test accuracy is {}'.format(accuracy_score(test.tags[i], prediction)))
