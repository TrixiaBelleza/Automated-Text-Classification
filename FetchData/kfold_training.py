'''
Computes the average FSCORE of a model from KFOLD Cross Vailidation
'''

import pymysql
import pandas as pd
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
# nltk.download('stopwords')
stop_words =  nltk.corpus.stopwords.words('english')
new_stop_words = ['(', ')', '[', ']', '{', '}', '"', "'", '``', '""',"''", ',', '.', '“', '”', '’', '`']
stop_words.extend(new_stop_words)

# # Connect to the database
db_connection = pymysql.connect(host='localhost',
                             user='root',
                             password='592008',
                             db='questions_db',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
categories = ['python', 'javascript', 'java', 'c', 'r', 'while_loop', 'for_loop']
df = pd.read_sql('SELECT * FROM train_data', con=db_connection)
# print(df)
db_connection.close()

kf = KFold(n_splits=10)

ave_f1scores = {}
for train_index, test_index in kf.split(df):
    print("TRAIN:", train_index, "TEST:", test_index)
    
    train = df.iloc[train_index]
    test = df.iloc[test_index]
    train_text = train['question_body']
    test_text = test['question_body']
    vectorizer = TfidfVectorizer(strip_accents='unicode', stop_words=stop_words, analyzer='word', ngram_range=(1,3), norm='l2', min_df=10)
    vectorizer.fit(train_text)
    x_train = vectorizer.transform(train_text)
    y_train = train.drop(labels = ['question_body'], axis=1)

    x_test = vectorizer.transform(test_text)
    y_test = test.drop(labels = ['question_body'], axis=1)

    # Using pipeline for applying linearSVC and one vs rest classifier
    SVC_pipeline = Pipeline([
                    ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
                ])
    # Using pipeline for applying logistic regression and one vs rest classifier
    LogReg_pipeline = Pipeline([
                    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
                ])
    # Using pipeline for applying Gaussian Naive Bayes and one vs rest classifier
    NB_pipeline = Pipeline([
                    ('clf', OneVsRestClassifier(MultinomialNB(), n_jobs=1)),
                ])
    for category in categories:
        print('... Processing {}'.format(category))

        # train the SVC model using X_dtm & y
        SVC_pipeline.fit(x_train, train[category])

        # compute the testing accuracy of SVC
        svc_prediction = SVC_pipeline.predict(x_test)
        print("SVC Prediction:")
        print(svc_prediction)
        print('Test F-SCORE is {}'.format(f1_score(test[category], svc_prediction)))
        print("\n")
        key_name = "svc_" + category
        if key_name in ave_f1scores:
            ave_f1scores[key_name] += f1_score(test[category], svc_prediction)
        else :
            ave_f1scores[key_name] = f1_score(test[category], svc_prediction)

        # Training logistic regression model on train data
        LogReg_pipeline.fit(x_train, train[category])
        # calculating test accuracy
        logreg_prediction = LogReg_pipeline.predict(x_test)
        print("LogReg Prediction:")
        print(logreg_prediction)
        print('Test F-SCORE is {}'.format(f1_score(test[category], logreg_prediction)))
        print("\n")
        key_name = "logreg_" + category
        if key_name in ave_f1scores:
            ave_f1scores[key_name] += f1_score(test[category], logreg_prediction)
        else :
            ave_f1scores[key_name] = f1_score(test[category], logreg_prediction)
        # Training logistic regression model on train data
        NB_pipeline.fit(x_train, train[category])
        # calculating test accuracy
        nb_prediction = NB_pipeline.predict(x_test)
        print("NB Prediction:")
        print(nb_prediction)
        print('Test F-SCORE is {}'.format(f1_score(test[category], nb_prediction)))
        print("\n")
        key_name = "nb_" + category
        if key_name in ave_f1scores:
            ave_f1scores[key_name] += f1_score(test[category], nb_prediction)
        else :
            ave_f1scores[key_name] = f1_score(test[category], nb_prediction)
print("Average f1 Scores:")

ave_f1scores = {k: v / 10 for k, v in ave_f1scores.items()}
print(ave_f1scores)