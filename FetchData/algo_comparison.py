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
df = pd.read_sql('SELECT * FROM clean_train_data', con=db_connection)
db_connection.close()

train, test = train_test_split(df, random_state=42, test_size=0.20, shuffle=True)

train_text = train['question_body']
test_text = test['question_body']

vectorizer = TfidfVectorizer(strip_accents='unicode', stop_words=stop_words, analyzer='word', ngram_range=(1,3), norm='l2')
vectorizer.fit(train_text)
# pickle.dump(vectorizer, open('vectorizer.sav', 'wb'))


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

predicted_list = []
for i in range(len(test)):
    predicted = {'question_body' : '', 'predicted_tags' : []}
    predicted['question_body'] = test["question_body"].values[i]
    predicted_list.append(predicted)

# Store models of each category
# filename = <model name> + <category name> + '.sav'
for category in categories:
    print('... Processing {}'.format(category))

    # train the SVC model using X_dtm & y
    SVC_pipeline.fit(x_train, train[category])

    # #Save dataframe
    # df_filename = category + '-df.png'

    # compute the testing accuracy of SVC
    svc_prediction = SVC_pipeline.predict(x_test)
    print("SVC Prediction:")
    print(svc_prediction)
    print('Test F-SCORE is {}'.format(f1_score(test[category], svc_prediction)))
    print("\n")
    print('Test ACCURACY is {}'.format(accuracy_score(test[category], svc_prediction)))
    print("\n")
    print('Test RECALL is {}'.format(recall_score(test[category], svc_prediction)))
    print("\n")
    print('Test PRECISION is {}'.format(precision_score(test[category], svc_prediction)))


    # Training logistic regression model on train data
    LogReg_pipeline.fit(x_train, train[category])
    # calculating test accuracy
    logreg_prediction = LogReg_pipeline.predict(x_test)
    print("LogReg Prediction:")
    print(logreg_prediction)
    print('Test F-SCORE is {}'.format(f1_score(test[category], logreg_prediction)))
    print("\n")
    print('Test ACCURACY is {}'.format(accuracy_score(test[category], logreg_prediction)))
    print("\n")
    print('Test RECALL is {}'.format(recall_score(test[category], logreg_prediction)))
    print("\n")
    print('Test PRECISION is {}'.format(precision_score(test[category], logreg_prediction)))

    # Training logistic regression model on train data
    NB_pipeline.fit(x_train, train[category])
    # calculating test accuracy
    nb_prediction = NB_pipeline.predict(x_test)
    print("NB Prediction:")
    print(nb_prediction)
    print('Test F-SCORE is {}'.format(f1_score(test[category], nb_prediction)))
    print("\n")
    print('Test ACCURACY is {}'.format(accuracy_score(test[category], nb_prediction)))
    print("\n")
    print('Test RECALL is {}'.format(recall_score(test[category], nb_prediction)))
    print("\n")
    print('Test PRECISION is {}'.format(precision_score(test[category], nb_prediction)))

    df_results = np.array([['','Recall','Precision', 'F-score', 'Accuracy'],
                ['SVC',recall_score(test[category], svc_prediction), precision_score(test[category], svc_prediction), f1_score(test[category], svc_prediction), accuracy_score(test[category], svc_prediction)],
                ['LogisticRegression',recall_score(test[category], logreg_prediction), precision_score(test[category], logreg_prediction), f1_score(test[category], logreg_prediction), accuracy_score(test[category], logreg_prediction)],
                ['NaiveBayes',recall_score(test[category], nb_prediction), precision_score(test[category], nb_prediction), f1_score(test[category], nb_prediction), accuracy_score(test[category], nb_prediction)]])

    df = pd.DataFrame(data=df_results[1:,1:],
                  index=df_results[1:,0],
                  columns=df_results[0,1:])
    print(df)

    # ax = plt.subplot(111, frame_on=False) # no visible frame
    fig, ax = plt.subplots(figsize=(12, 2)) # set size frame
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis
    ax.axis('off')
    table(ax, df, loc='center', colWidths=[0.17]*len(df.columns))  # where df is your data frame

    plt.savefig('mytable' + category +'.png')