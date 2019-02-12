import pickle
import re
import html2text

import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.stem.snowball import SnowballStemmer

stop_words =  nltk.corpus.stopwords.words('english')
new_stop_words = ['(', ')', '[', ']', '{', '}', '"', "'", '``', '""',"''", ',', '.', '“', '”', '’', '`']
stop_words.extend(new_stop_words)

stemmer = SnowballStemmer("english")

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

# print(questions["items"])
h = html2text.HTML2Text()
h.ignore_links = True

categories = ['python', 'javascript', 'java', 'c', 'r', 'while_loop', 'for_loop']

#Get new text data
text_list = []
text = input("Enter text: ")
# text = [' i have a list consist of like 20000 lists. i use each list 3rd element as a flag. i want to do some oper on this list as long as at least one element flag is 0 it like: my_list = [["a" "b" 0] ["c" "d" 0] ["e" "f" 0] .....] in the begin all flag are 0. i use a while loop to check if at least one element flag is 0: def check(list_): for item in list_: if item[2] == 0: return true return fals if `check(my_list)` return `true` then i continu work on my list: while check(my_list): for item in my_list: if condition: item[2] = 1 else: do_sth() actual i want to remov an element in my_list as i iter over it but im not allow to remov item as i iter over it. origin my_list didnt have flags: my_list = [["a" "b"] ["c" "d"] ["e" "f"] .....] sinc i couldnt remov element as i iter over it i invent these flags. but the `my_list` contain mani item and `while` loop read all of them at each `for` loop and it consum lot of time do you have ani suggest']
text = cleanPunc(text.lower())
text = stemming(text)

text_list.append(text)

vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
x_text = vectorizer.transform(text_list)

predicted_tags = []
for category in categories:
    print('... Processing {}'.format(category))

    #load model
    filename = 'svc-' + category + '.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.predict(x_text)
    print(result)

    if result[0] == 1:
        predicted_tags.append(category)
print(predicted_tags)
'''
#Testcases:
I'm using the python logging module, along with python-json-logger and I would like to add a few key {"app_name": "myapp", "env": "prod"} To all of my logs automatically without doing the following. logger.info("Something happened", extra={"app_name": "myapp", "env": "prod"}) But for it to work as if I had. :)
'''