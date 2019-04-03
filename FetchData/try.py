import datapreprocessing 

question_body = "how to code in c"
question_body = datapreprocessing.cleanPunc(question_body)
question_body = datapreprocessing.stemming(question_body)	
print(question_body)