import json
import re

import requests
from bs4 import BeautifulSoup


# 获取微信读书全局js存储模块
def getHomeStoreModule():
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
    return dict_data['homeStoreModule']
