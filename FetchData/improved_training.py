import pymysql
import pandas as pd
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
# Connect to the database
db_connection = pymysql.Connect(host='localhost',
                             user='root',
                             password='592008',
                             db='questions_db',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
# categories = ['python/', 'javascript', 'java', 'c', 'r', 'mysql', 'html/', 'if_statement', 'while_loop', 'for_loop', 'css']
pickle_in = open("categories_dict.pickle","rb")
categories = pickle.load(pickle_in)

df = pd.read_sql('SELECT * FROM complete_train_data2', con=db_connection)
db_connection.close()

train, test = train_test_split(df, train_size=0.8, test_size=0.20, shuffle=True)

train_text = train['question_body']

test_text = test['question_body']

# vectorizer = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode', stop_words=stop_words, analyzer='word', ngram_range=(1,2), norm='l2', min_df=10)
vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=10, norm='l2', encoding='latin-1', ngram_range=(1, 3))
vectorizer.fit_transform(train_text)
pickle.dump(vectorizer, open('../extension-app/models/vectorizer.sav', 'wb'))

x_train = vectorizer.transform(train_text)
y_train = train.drop(labels = ['question_body'], axis=1)

x_test = vectorizer.transform(test_text)
y_test = test.drop(labels = ['question_body'], axis=1)

# Using pipeline for applying linearSVC and one vs rest classifier
SVC_pipeline = Pipeline([
                ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=-1)),
            ])

for category in categories:
    print('... Processing {}'.format(category))

    # train the SVC model using X_dtm & y
    SVC_pipeline.fit(x_train, train[category])

    # Store models of each category
    # filename = <model name> + <category name> + '.sav'
    filename = 'svc-' + category + '.sav'
    pickle.dump(SVC_pipeline, open('../extension-app/models/' + filename, 'wb'))
    # compute the testing accuracy of SVC
    svc_prediction = SVC_pipeline.predict(x_test)
    print("SVC Prediction:")
    print(svc_prediction)
    print('Test macro F-SCORE is {}'.format(f1_score(test[category], svc_prediction, average='macro')))

    print('Test micro F-SCORE is {}'.format(f1_score(test[category], svc_prediction, average='micro')))


    print('Test F-SCORE is {}'.format(f1_score(test[category], svc_prediction)))
    print('Test Accuracy is {}'.format(accuracy_score(test[category], svc_prediction)))
    print("\n")
   