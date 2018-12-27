import json
import html2text
import math
import nltk
import re #regEx
import pandas as pd
from scipy import sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

training_data = []
categories = ['python', 'if-statement', 'for-loop', 'java']

for i in range(len(data["items"])):
	body_tag_dict = {'question_body' : '', 'python' : 0, 'if-statement': 0, 'for-loop': 0, 'java' : 0}
	#Convert html code to text since data["items"][i]["body"] returns something like this: "<p>I have 2 columns <p>""
	html_to_text = h.handle(data["items"][i]["body"])
	html_to_text = html_to_text.lower()
	clean_text = preprocess_text(html_to_text)

	body_tag_dict['question_body'] = clean_text
	for j in range(len(data["items"][i]["tags"])):
		if data["items"][i]["tags"][j] in categories:
			current_key_index = data["items"][i]["tags"][j]
			body_tag_dict[current_key_index] = 1
		
	training_data.append(body_tag_dict)
print("training_data")
print(training_data)
df = pd.DataFrame(training_data)
# print(df)
train, test = train_test_split(df, random_state=42, test_size=0.20, shuffle=True)

train_text = train['question_body']
test_text = test['question_body']

vectorizer = TfidfVectorizer(strip_accents='unicode', stop_words=stop_words, analyzer='word', ngram_range=(1,3), norm='l2')
vectorizer.fit(train_text)
vectorizer.fit(test_text)

x_train = vectorizer.transform(train_text)
y_train = train.drop(labels = ['question_body'], axis=1)

x_test = vectorizer.transform(test_text)
y_test = test.drop(labels = ['question_body'], axis=1)

print("TEST")
print(test)

# Using pipeline for applying linearSVC and one vs rest classifier
SVC_pipeline = Pipeline([
				('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
			])
for category in categories:
	print('... Processing {}'.format(category))
	
	# train the model using X_dtm & y

	SVC_pipeline.fit(x_train, train[category])
	# print("SVC Classes")
	# print(SVC_pipeline.classes_)
	# compute the testing accuracy
	prediction = SVC_pipeline.predict(x_test)
	print("prediction:")
	print(prediction)

	print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
