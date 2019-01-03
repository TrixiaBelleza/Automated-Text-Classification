import requests, constants, utils, time
from bs4 import BeautifulSoup
from getpass import getpass

#Constants 
url = constants.Url()
session_req = requests.session()

credentials = {
	'email' : '',
	'password' : '',
	'fkey' : '5ed96a23aa7df56c8c8fe0e3bbdfb01a8e33890d211a921b77358e6147099376'
}

#Login
credentials['email'] = input("Email: ")
credentials['password'] = getpass("Password: ")

print("Logging in.........")

res = session_req.post(
	url.LOGIN,
	data = credentials,
	headers = dict(referer = url.LOGIN)
)

utils.printStatus(res);

#Connect to ask a question page
print("Ask a Question page")
url = 'https://meta.stackoverflow.com/questions/ask'

res = session_req.get(
	url,
	headers = dict(referer = url)
)
utils.printStatus(res)

#Get data
# Getting course titles
soup = BeautifulSoup(res.content, features='lxml')
scraped = soup.findAll('div', attrs = {'id': 'wmd-preview'})

# while True:

# 	print("Scraped:")
# 	print(scraped)
# 	time.sleep(8)