import pickle
import re
from nltk.stem.snowball import SnowballStemmer
from flask import Flask, render_template, request, jsonify, json
from flask_cors import CORS, cross_origin
import time

stemmer = SnowballStemmer("english")
text_list = []
categories = ['python', 'javascript', 'java', 'c', 'r', 'mysql', 'html', 'if_statement', 'while_loop', 'for_loop', 'css']
model = {}

for category in categories:
       #load model
        filename = 'svc-' + category + '.sav'
        loaded_model = pickle.load(open('/home/trixiabells/deploy/' + filename, 'rb'))
        model[category] = loaded_model

def cleanPunc(sentence): #function to clean the word of unnecessary punctuation or special characters using re library or regex
    cleaned = re.sub(r'[?|!|,|~|^]',r'',sentence)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
vectorizer = pickle.load(open('/home/trixiabells/deploy/vectorizer.sav', 'rb'))

@app.route('/')
def index():
    return '<h1> Deployed! </h1>'

@app.route('/_get_text_input/', methods=['POST'])
def get_text_input():

    #Get new text data
    text_list = []
    text = request.get_data().decode("utf-8")
    text = cleanPunc(text.lower())
    text = stemming(text)
    text_list.append(text)

    vectorized_text = vectorizer.transform(text_list)

    predicted_tags = []
    for category in categories:
        model_category = model.get(category)
        result = model_category.predict(vectorized_text)
        if result[0] == 1:
            if category == "while_loop":
                predicted_tags.append('while-loop')
            elif category == "for_loop":
                predicted_tags.append('for-loop')
            elif category == "if_statement":
                predicted_tags.append('if-statement')
            else:
                predicted_tags.append(category)
    if len(predicted_tags) == 0:
        predicted_tags.append('others')
    return jsonify({'predicted_tags' : predicted_tags})

if __name__ == "__main__":
	app.run(debug=True, threaded=True)
