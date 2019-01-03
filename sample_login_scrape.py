import requests
from lxml import html

EMAIL = "animetotemodaisuki@gmail.com"
PASSWORD = "tqbfjotld1"

LOGIN_URL = "https://stackoverflow.com/users/login?ssrc=anon_ask&returnurl=https%3a%2f%2fstackoverflow.com%2fquestions%2fask%3fwizard%3d1"
URL = "https://stackoverflow.com/questions/ask?wizard=1"

def main():
    session_requests = requests.session()

    # Get login csrf token
    result = session_requests.get(LOGIN_URL)
    tree = html.fromstring(result.text)
    print("Tree")
    print(tree)
    authenticity_token = list(set(tree.xpath("//input[@name='fkey']/@value")))[0]
    print(authenticity_token)
    
    # Create payload
    payload = {
        "email": EMAIL, 
        "password": PASSWORD, 
        "fkey": authenticity_token
    }

    # # Perform login
    result = session_requests.post(LOGIN_URL, data = payload, headers = dict(referer = LOGIN_URL))
    print("result")
    print(result)
    # Scrape url
    result = session_requests.get(URL, headers = dict(referer = URL))
    tree = html.fromstring(result.content)
    print("tree2:")
    print(tree)

    page = tree.xpath('//div[@class = "grid grid__center h100"]')
    print(page)

    # bucket_names = tree.xpath("//div[@class='repo-list--repo']/a/text()")

    # print(bucket_names)

if __name__ == '__main__':
    main()