import time
from multiprocessing.dummy import Pool

import requests


def query(url):
    requests.get(url)


def oneProcess():
    start = time.time()
    for i in range(50):
        query('https://baidu.com')
    end = time.time()
    print(f'单线程循环访问100次百度首页，耗时：{end - start}')



def multiplyHandle(method,url_list):
    pool = Pool(5)
    pool.map(method, url_list)

