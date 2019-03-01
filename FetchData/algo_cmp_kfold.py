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

nltk.download('stopwords')
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
categories = ['python', 'javascript', 'java', 'c', 'r', 'mysql', 'html', 'if_statement', 'while_loop', 'for_loop']
df = pd.read_sql('SELECT * FROM complete_train_data', con=db_connection)
db_connection.close()

kf = KFold(n_splits=10)

ave_f1scores = {}
ave_recall = {}
ave_precision = {}
ave_accuracy = {}

for train_index, test_index in kf.split(df):
    print("TRAIN:", train_index, "TEST:", test_index)

    train, test = train_test_split(df, random_state=42, test_size=0.20, shuffle=True)

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
                    ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
                ])
    # Using pipeline for applying logistic regression and one vs rest classifier
    LogReg_pipeline = Pipeline([
                    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
                ])
    # Using pipeline for applying Gaussian Naive Bayes and one vs rest classifier
    NB_pipeline = Pipeline([
                    ('clf', OneVsRestClassifier(MultinomialNB(alpha=5), n_jobs=1)),
                ])

    predicted_list = []
    for i in range(len(test)):
        predicted = {'question_body' : '', 'predicted_tags' : []}
        predicted['question_body'] = test["question_body"].values[i]
        predicted_list.append(predicted)


    for category in categories:
        print('... Processing {}'.format(category))

        # train the SVC model using X_dtm & y
        SVC_pipeline.fit(x_train, train[category])

        # compute the testing accuracy of SVC
        svc_prediction = SVC_pipeline.predict(x_test)
        print("SVC Prediction:")
        print(svc_prediction)
        print('Test F-SCORE is {}'.format(f1_score(test[category], svc_prediction, average='macro')))
        # svc_fscores += (f1_score(test[category], svc_prediction),)
        print("\n")
        print('Test ACCURACY is {}'.format(accuracy_score(test[category], svc_prediction)))
        print("\n")
        print('Test RECALL is {}'.format(recall_score(test[category], svc_prediction, average='macro')))
        print("\n")
        print('Test PRECISION is {}'.format(precision_score(test[category], svc_prediction, average='macro')))

        # SVC F1 Score
        key_name = "svc_" + category
        if key_name in ave_f1scores:
            ave_f1scores[key_name] += f1_score(test[category], svc_prediction, average='macro')
        else :
            ave_f1scores[key_name] = f1_score(test[category], svc_prediction, average='macro')

        # SVC Recall 
        key_name = "svc_" + category
        if key_name in ave_recall:
            ave_recall[key_name] += recall_score(test[category], svc_prediction, average='macro')
        else :
            ave_recall[key_name] = recall_score(test[category], svc_prediction, average='macro')

        # SVC Precision
        key_name = "svc_" + category
        if key_name in ave_precision:
            ave_precision[key_name] += precision_score(test[category], svc_prediction, average='macro')
        else :
            ave_precision[key_name] = precision_score(test[category], svc_prediction, average='macro')

        # SVC Accuracy
        key_name = "svc_" + category
        if key_name in ave_accuracy:
            ave_accuracy[key_name] += accuracy_score(test[category], svc_prediction)
        else :
            ave_accuracy[key_name] = accuracy_score(test[category], svc_prediction)

        # Training logistic regression model on train data
        LogReg_pipeline.fit(x_train, train[category])
        # calculating test accuracy
        logreg_prediction = LogReg_pipeline.predict(x_test)
        print("LogReg Prediction:")
        print(logreg_prediction)
        print('Test F-SCORE is {}'.format(f1_score(test[category], logreg_prediction, average='macro')))
        # logreg_fscores += (f1_score(test[category], logreg_prediction),)
        print("\n")
        print('Test ACCURACY is {}'.format(accuracy_score(test[category], logreg_prediction)))
        print("\n")
        print('Test RECALL is {}'.format(recall_score(test[category], logreg_prediction, average='macro')))
        print("\n")
        print('Test PRECISION is {}'.format(precision_score(test[category], logreg_prediction, average='macro')))

        #Log Reg F1 Score
        key_name = "logreg_" + category
        if key_name in ave_f1scores:
            ave_f1scores[key_name] += f1_score(test[category], logreg_prediction, average='macro')
        else :
            ave_f1scores[key_name] = f1_score(test[category], logreg_prediction, average='macro')

        # LogReg Recall 
        key_name = "logreg_" + category
        if key_name in ave_recall:
            ave_recall[key_name] += recall_score(test[category], logreg_prediction, average='macro')
        else :
            ave_recall[key_name] = recall_score(test[category], logreg_prediction, average='macro')

        # logreg Precision
        key_name = "logreg_" + category
        if key_name in ave_precision:
            ave_precision[key_name] += precision_score(test[category], logreg_prediction, average='macro')
        else :
            ave_precision[key_name] = precision_score(test[category], logreg_prediction, average='macro')

        # logreg Accuracy
        key_name = "logreg_" + category
        if key_name in ave_accuracy:
            ave_accuracy[key_name] += accuracy_score(test[category], logreg_prediction)
        else :
            ave_accuracy[key_name] = accuracy_score(test[category], logreg_prediction)

        # Training logistic regression model on train data
        NB_pipeline.fit(x_train, train[category])
        # calculating test accuracy
        nb_prediction = NB_pipeline.predict(x_test)
        print("NB Prediction:")
        print(nb_prediction)
        print('Test F-SCORE is {}'.format(f1_score(test[category], nb_prediction, average='macro')))
        # nb_fscores += (f1_score(test[category], nb_prediction),)
        print("\n")
        print('Test ACCURACY is {}'.format(accuracy_score(test[category], nb_prediction)))
        print("\n")
        print('Test RECALL is {}'.format(recall_score(test[category], nb_prediction, average='macro')))
        print("\n")
        print('Test PRECISION is {}'.format(precision_score(test[category], nb_prediction, average='macro')))

        #NB F1 Score
        key_name = "nb_" + category
        if key_name in ave_f1scores:
            ave_f1scores[key_name] += f1_score(test[category], nb_prediction, average='macro')
        else :
            ave_f1scores[key_name] = f1_score(test[category], nb_prediction, average='macro')

        # nb Recall 
        key_name = "nb_" + category
        if key_name in ave_recall:
            ave_recall[key_name] += recall_score(test[category], nb_prediction, average='macro')
        else :
            ave_recall[key_name] = recall_score(test[category], nb_prediction, average='macro')

        # nb Precision
        key_name = "nb_" + category
        if key_name in ave_precision:
            ave_precision[key_name] += precision_score(test[category], nb_prediction, average='macro')
        else :
            ave_precision[key_name] = precision_score(test[category], nb_prediction, average='macro')

        # nb Accuracy
        key_name = "nb_" + category
        if key_name in ave_accuracy:
            ave_accuracy[key_name] += accuracy_score(test[category], nb_prediction)
        else :
            ave_accuracy[key_name] = accuracy_score(test[category], nb_prediction)

print("Average f1 Scores:")

ave_f1scores = {k: v / 10 for k, v in ave_f1scores.items()}
print(ave_f1scores)

print("Average recall Scores:")

ave_recall = {k: v / 10 for k, v in ave_recall.items()}
print(ave_recall)

print("Average precision Scores:")

ave_precision = {k: v / 10 for k, v in ave_precision.items()}
print(ave_precision)

print("Average accuracy Scores:")

ave_accuracy = {k: v / 10 for k, v in ave_accuracy.items()}
print(ave_accuracy)

print(ave_accuracy.get('svc_python'))

svc_fscores = ()
logreg_fscores = ()
nb_fscores = ()
for category in categories: 
    df_results = np.array([['','Recall','Precision', 'F-score', 'Accuracy'],
                ['SVC',ave_recall.get('svc_'+category), ave_precision.get('svc_'+category), ave_f1scores.get('svc_'+category), ave_accuracy.get('svc_'+category)],
                ['LogisticRegression',ave_recall.get('logreg_'+category), ave_precision.get('logreg_'+category), ave_f1scores.get('logreg_'+category), ave_accuracy.get('logreg_'+category)],
                ['NaiveBayes',ave_recall.get('nb_'+category), ave_precision.get('nb_'+category), ave_f1scores.get('nb_'+category), ave_accuracy.get('nb_'+category)]])

    dataframe = pd.DataFrame(data=df_results[1:,1:],
                  index=df_results[1:,0],
                  columns=df_results[0,1:])
    fig, ax = plt.subplots(figsize=(12, 2)) # set size frame
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis
    ax.axis('off')
    table(ax, dataframe, loc='center', colWidths=[0.17]*len(dataframe.columns))  # where df is your data frame

    plt.savefig('FARP_' + category +'.png')
    svc_fscores += (ave_f1scores.get('svc_'+category),)
    logreg_fscores += (ave_f1scores.get('logreg_'+category),)
    nb_fscores += (ave_f1scores.get('nb_'+category),)

# create plot for F-scores for each category.
n_groups = 10
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.3
opacity = 0.8

 
rects1 = plt.bar(index, svc_fscores, bar_width,
alpha=opacity,
color='b',
label='SVC')

rects2 = plt.bar(index + bar_width, logreg_fscores, bar_width,
alpha=opacity,
color='g',
label='LogReg')

rects3 = plt.bar(index + bar_width + bar_width, nb_fscores, bar_width,
alpha=opacity,
color='y',
label='NB')

categories_tuple = ('Python', 'JS', 'Java', 'C', 'R', 'MySQL', 'HTML', 'If', 'While', 'For')
plt.xlabel('Tags')
plt.ylabel('F Scores')
plt.title('F Scores by tags')
plt.xticks(index + bar_width, categories_tuple)
plt.legend()
 
plt.tight_layout()
plt.savefig('avefscores.png')