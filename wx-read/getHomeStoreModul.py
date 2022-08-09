import requests
from bs4 import BeautifulSoup
import json
import re

def getHomeStoreModulDict():
    response = requests.get('https://weread.qq.com/web/category/')
    begin = 'window.__INITIAL_STATE__='
    end = ';(function'
    soup = BeautifulSoup(response.text, features='html.parser')
    a = soup.select('script')
    r = a[0].text
    pattern = re.compile('.*{.*;\(function', re.S)
    result = re.match(pattern, r)
    text = result.group()[len(begin):-len(end)]
    dict_data = json.loads(text)
    # print(dict_data['homeStoreModule'])
    return dict_data['homeStoreModule']
