# Automated-Text-Classification
An automated text classification using machine learning approaches such as a Multi-label One-vs-Rest scheme and a Support Vector Machine

The FetchData directory contains:
algo_cmp_kfold.py (For graphs, statistics, and algorithm comparisons)
createTable.py (Code in creating table in MySQL)
insert2_data.py (Code for inserting data to MySQL)
load_model.py (Code for predicting, this was used as test code only. This was not used in the actual app.)
train_data.py (Code for training the data and producing models)

The extension-app directory contains:
contentScript.js (Main APP)
server.py (MAIN LOCAL SERVER)
server2.py (Used as testing server, not used in the actual app.)

To Start Training:
1. Store items to database
	*** createTable.py
	*** insert2_data.py
2. Run train_data.py
