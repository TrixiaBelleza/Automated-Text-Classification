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
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore') 
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
categories = ['python', 'javascript', 'java', 'c', 'r', 'mysql', 'html', 'if_statement', 'while_loop', 'for_loop', 'css']
df = pd.read_sql('SELECT * FROM complete_train_data2', con=db_connection)
db_connection.close()

train, test = train_test_split(df, random_state=42, train_size=0.80, test_size=0.20, shuffle=True)

train_text = train['question_body']
test_text = test['question_body']

vectorizer = TfidfVectorizer(strip_accents='unicode', stop_words=stop_words, analyzer='word', ngram_range=(1,3), norm='l2', min_df=15)
vectorizer.fit(train_text)
# pickle.dump(vectorizer, open('./data-models/vectorizer.sav', 'wb'))

x_train = vectorizer.transform(train_text)
y_train = train.drop(labels = ['question_body'], axis=1)

x_test = vectorizer.transform(test_text)
y_test = test.drop(labels = ['question_body'], axis=1)

# # Using pipeline for applying linearSVC and one vs rest classifier
SVC_pipeline = Pipeline([
                ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=-1)),
            ])
# Using pipeline for applying logistic regression and one vs rest classifier
LogReg_pipeline = Pipeline([
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),
            ])
# Using pipeline for applying Gaussian Naive Bayes and one vs rest classifier
NB_pipeline = Pipeline([
                ('clf', OneVsRestClassifier(MultinomialNB(), n_jobs=-1)),
            ])

predicted_list = []
for i in range(len(test)):
    predicted = {'question_body' : '', 'predicted_tags' : []}
    predicted['question_body'] = test["question_body"].values[i]
    predicted_list.append(predicted)

# Store models of each category
# filename = <model name> + <category name> + '.sav'
param_grid = {'clf__estimator__C': np.arange(1,5), 
                'clf__estimator__tol': [1, 0.01, 0.001, 0.0001, 0.00000001]}
nb_param_grid = {'clf__estimator__alpha': [1, 1e-1, 1e-2]}

svc_scores = {}
lr_scores = {}
nb_scores = {}
for category in categories:
    
    print('... Processing {}'.format(category))
    svc_clf_cv = GridSearchCV(SVC_pipeline, param_grid, cv=10, scoring='f1_macro')
    svc_clf_cv.fit(x_train, train[category])
    print("SVC:")

    print("Tuned Parameters: {}".format(svc_clf_cv.best_params_))
    print("Best score is: {}".format(svc_clf_cv.best_score_))

    lr_clf_cv = GridSearchCV(LogReg_pipeline, param_grid, cv=10, scoring='f1_macro')
    lr_clf_cv.fit(x_train, train[category])
    print("Log Reg:")

    print("Tuned Parameters: {}".format(lr_clf_cv.best_params_))
    print("Best score is: {}".format(lr_clf_cv.best_score_))


    nb_clf_cv = GridSearchCV(NB_pipeline, nb_param_grid, cv=10, scoring='f1_macro')
    nb_clf_cv.fit(x_train, train[category])
    print("NB:")

    print("Tuned Parameters: {}".format(nb_clf_cv.best_params_))
    print("Best score is: {}".format(nb_clf_cv.best_score_))


    SVC2_pipeline = Pipeline([
                ('clf', OneVsRestClassifier(LinearSVC(C=svc_clf_cv.best_params_.get('clf__estimator__C'), tol = svc_clf_cv.best_params_.get('clf__estimator__tol')), n_jobs=-1)),
            ])
    SVC2_pipeline.fit(x_train, train[category])

    # Store models of each category
    # filename = <model name> + <category name> + '.sav'
    filename = '2svc-' + category + '.sav'
    # pickle.dump(SVC2_pipeline, open('./data-models/' + filename, 'wb'))

    svc_prediction = SVC2_pipeline.predict(x_test)
    print("SVC Prediction:")
    # print(svc_prediction)
    print('Test macro F-SCORE is {}'.format(f1_score(test[category], svc_prediction, average='macro')))

    # Using pipeline for applying logistic regression and one vs rest classifier
    LogReg2_pipeline = Pipeline([
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag', C=lr_clf_cv.best_params_.get('clf__estimator__C'), tol = lr_clf_cv.best_params_.get('clf__estimator__tol')), n_jobs=-1)),
            ])
    LogReg2_pipeline.fit(x_train, train[category])
    # Store models of each category
    # filename = <model name> + <category name> + '.sav'
    filename = '2lr-' + category + '.sav'
    # pickle.dump(LogReg2_pipeline, open('./data-models/' + filename, 'wb'))

    lr_prediction = LogReg2_pipeline.predict(x_test)
    print("LR Prediction:")
    # print(lr_prediction)
    print('Test macro F-SCORE is {}'.format(f1_score(test[category], lr_prediction, average='macro')))
    

    # Using pipeline for applying logistic regression and one vs rest classifier
    NB2_pipeline = Pipeline([
                ('clf', OneVsRestClassifier(MultinomialNB(alpha=nb_clf_cv.best_params_.get('clf__estimator__alpha')), n_jobs=-1)),
            ])
    NB2_pipeline.fit(x_train, train[category])
    # Store models of each category
    # filename = <model name> + <category name> + '.sav'
    filename = '2nb-' + category + '.sav'
    # pickle.dump(NB2_pipeline, open('./data-models/' + filename, 'wb'))

    nb_prediction = NB2_pipeline.predict(x_test)
    # print("nb Prediction:")
    # print(nb_prediction)
    print('Test macro F-SCORE is {}'.format(f1_score(test[category], nb_prediction, average='macro')))
    

