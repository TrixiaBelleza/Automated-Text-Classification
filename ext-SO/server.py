import pickle
import re
from nltk.stem.snowball import SnowballStemmer
from flask import Flask, render_template, request, jsonify, json
from flask_cors import CORS, cross_origin

stemmer = SnowballStemmer("english")
text_list = []
categories = ['python', 'javascript', 'java', 'c', 'r', 'while_loop', 'for_loop']

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

@app.route('/_get_text_input/', methods=['POST'])
def get_text_input():
    #Get new text data
    text_list = []
    text = request.get_data().decode("utf-8")
    text = cleanPunc(text.lower())
    text = stemming(text)
    text_list.append(text)

    print(text_list)
    vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
    vectorized_text = vectorizer.transform(text_list)

    predicted_tags = []
    for category in categories:
       #load model
        filename = 'svc-' + category + '.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        result = loaded_model.predict(vectorized_text)
        if result[0] == 1:
            predicted_tags.append(category)
    if len(predicted_tags) == 0:
        predicted_tags = {'none'}
    print(predicted_tags)

    return jsonify({'data': render_template('response.html', predicted_tags=predicted_tags)})

if __name__ == "__main__":
	app.run(debug=True, threaded=True)
