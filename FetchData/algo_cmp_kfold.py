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

def create_10fold_df(fold_svc_score_tuple, fold_logreg_score_tuple, fold_nb_score_tuple, score_type):
    score_df_results = np.array([['','SVM','LR', 'NB'],
                ['Fold1', fold_svc_score_tuple[0], fold_logreg_score_tuple[0], fold_nb_score_tuple[0]],
                ['Fold2', fold_svc_score_tuple[1], fold_logreg_score_tuple[1], fold_nb_score_tuple[1]]
                # ['Fold3', fold_svc_score_tuple[2], fold_logreg_score_tuple[2], fold_nb_score_tuple[2]],
                # ['Fold4', fold_svc_score_tuple[3], fold_logreg_score_tuple[3], fold_nb_score_tuple[3]],
                # ['Fold5', fold_svc_score_tuple[4], fold_logreg_score_tuple[4], fold_nb_score_tuple[4]],
                # ['Fold6', fold_svc_score_tuple[5], fold_logreg_score_tuple[5], fold_nb_score_tuple[5]],
                # ['Fold7', fold_svc_score_tuple[6], fold_logreg_score_tuple[6], fold_nb_score_tuple[6]],
                # ['Fold8', fold_svc_score_tuple[7], fold_logreg_score_tuple[7], fold_nb_score_tuple[7]],
                # ['Fold9', fold_svc_score_tuple[8], fold_logreg_score_tuple[8], fold_nb_score_tuple[8]],
                # ['Fold10', fold_svc_score_tuple[9], fold_logreg_score_tuple[9], fold_nb_score_tuple[9]]
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


# nltk.download('stopwords')
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

ave_f1scores = {}
ave_recall = {}
ave_precision = {}
ave_accuracy = {}
fold_svcf1scores_list = []
fold_svcrecall_list = []
fold_svcprecision_list = []
fold_svcaccuracy_list = []

fold_logregf1scores_list = []
fold_logregrecall_list = []
fold_logregprecision_list = []
fold_logregaccuracy_list = []

fold_nbf1scores_list = []
fold_nbrecall_list = []
fold_nbprecision_list = []
fold_nbaccuracy_list = []

for train_index, test_index in kf.split(df):
    fold_svcf1scores = 0
    fold_svcrecall = 0
    fold_svcprecision = 0
    fold_svcaccuracy = 0

    fold_logregf1scores = 0
    fold_logregrecall = 0
    fold_logregprecision = 0
    fold_logregaccuracy = 0

    fold_nbf1scores = 0
    fold_nbrecall = 0
    fold_nbprecision = 0
    fold_nbaccuracy = 0

    print("TRAIN:", train_index, "TEST:", test_index)

    train, test = train_test_split(df, random_state=42, train_size=0.67, test_size=0.33, shuffle=True)

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

    predicted_list = []
    for i in range(len(test)):
        predicted = {'question_body' : '', 'predicted_tags' : []}
        predicted['question_body'] = test["question_body"].values[i]
        predicted_list.append(predicted)

    svc_fscores = ()
    logreg_fscores = ()
    nb_fscores = ()
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
        print('Test ACCURACY is {}'.format(accuracy_score(test[category], svc_prediction)))
        fold_svcaccuracy += accuracy_score(test[category], svc_prediction)
        print("\n")
        print('Test RECALL is {}'.format(recall_score(test[category], svc_prediction, average='macro')))
        fold_svcrecall += recall_score(test[category], svc_prediction, average='macro')
        print("\n")
        print('Test PRECISION is {}'.format(precision_score(test[category], svc_prediction, average='macro')))
        fold_svcprecision += precision_score(test[category], svc_prediction, average='macro')
        
        # Training logistic regression model on train data
        LogReg_pipeline.fit(x_train, train[category])
        # calculating test accuracy
        logreg_prediction = LogReg_pipeline.predict(x_test)
        print("LogReg Prediction:")
        print(logreg_prediction)
        print('Test F-SCORE is {}'.format(f1_score(test[category], logreg_prediction, average='macro')))
        fold_logregf1scores += f1_score(test[category], logreg_prediction, average='macro')
        print("\n")
        print('Test ACCURACY is {}'.format(accuracy_score(test[category], logreg_prediction)))
        fold_logregaccuracy += accuracy_score(test[category], logreg_prediction)
        print("\n")
        print('Test RECALL is {}'.format(recall_score(test[category], logreg_prediction, average='macro')))
        fold_logregrecall += recall_score(test[category], logreg_prediction, average='macro')
        print("\n")
        print('Test PRECISION is {}'.format(precision_score(test[category], logreg_prediction, average='macro')))
        fold_logregprecision += precision_score(test[category], logreg_prediction, average='macro')

        # Training logistic regression model on train data
        NB_pipeline.fit(x_train, train[category])
        # calculating test accuracy
        nb_prediction = NB_pipeline.predict(x_test)
        print("NB Prediction:")
        print(nb_prediction)
        print('Test F-SCORE is {}'.format(f1_score(test[category], nb_prediction, average='macro')))
        fold_nbf1scores += f1_score(test[category], nb_prediction, average='macro')
        print("\n")
        print('Test ACCURACY is {}'.format(accuracy_score(test[category], nb_prediction)))
        fold_nbaccuracy += accuracy_score(test[category], nb_prediction)
        print("\n")
        print('Test RECALL is {}'.format(recall_score(test[category], nb_prediction, average='macro')))
        fold_nbrecall += recall_score(test[category], nb_prediction, average='macro')
        print("\n")
        print('Test PRECISION is {}'.format(precision_score(test[category], nb_prediction, average='macro')))
        fold_nbprecision += precision_score(test[category], nb_prediction, average='macro')

    # print(fold_svcf1scores)
    fold_svcf1scores = fold_svcf1scores/len(categories)
    fold_svcf1scores_list.append(fold_svcf1scores)
    print("Fold SVC FSCORES LIST")
    print(fold_svcf1scores_list)
    # fold_logregf1scores_tuple += (fold_logregf1scores/len(categories),)
    # fold_nbf1scores_tuple += (fold_nbf1scores/len(categories),)

    # fold_svcaccuracy = fold_svcaccuracy/len(categories)
    # fold_logregaccuracy = fold_logregaccuracy/len(categories)
    # fold_nbaccuracy = fold_nbaccuracy/len(categories)

    # fold_svcaccuracy_tuple += (fold_svcaccuracy/len(categories),)
    # fold_logregaccuracy_tuple += (fold_logregaccuracy/len(categories),)
    # fold_nbaccuracy_tuple += (fold_nbaccuracy/len(categories),)

    # fold_svcrecall_tuple += (fold_svcrecall/len(categories),)
    # fold_logregrecall_tuple += (fold_logregrecall/len(categories),)
    # fold_nbrecall_tuple += (fold_nbrecall/len(categories),)

    # fold_svcprecision_tuple += (fold_svcprecision/len(categories),)
    # fold_logregprecision_tuple += (fold_logregprecision/len(categories),)
    # fold_nbprecision_tuple += (fold_nbprecision/len(categories),)

# create_10fold_df(fold_svcf1scores_tuple, fold_logregf1scores_tuple, fold_nbf1scores_tuple, 'fscore')
# create_10fold_df(fold_svcaccuracy_tuple, fold_logregaccuracy_tuple, fold_nbaccuracy_tuple, 'acc')
# create_10fold_df(fold_svcrecall_tuple, fold_logregrecall_tuple, fold_nbrecall_tuple, 'recall')
# create_10fold_df(fold_svcprecision_tuple, fold_logregprecision_tuple, fold_nbprecision_tuple, 'precision')

# print("Average f1 Scores:")

