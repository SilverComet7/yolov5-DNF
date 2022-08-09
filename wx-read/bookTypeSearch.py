import time
import random
import requests
import csv
import os
from  getHomeStoreModul import getHomeStoreModulDict
from multiprocessing_test import  multiplyHandle
from multiprocessing.dummy import Pool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# 不重复，完整，中断要可续接，进数据库或者excel记录

# 书名
# 分类
# 最大免费章节  maxFreeChapter
# 推荐值   newRating
# 出版时间
# 介绍
# 图片
# 推荐值


def getWXBooks(booksId):

    bookList = []

    for i in range(554):
        page = i * 20
        rTime = random.randint(1, 3)  # 随机从1到3内取一个整数值
        time.sleep(rTime)  # 把随机取出的整数值传到等待函数中
        res = requests.get(
            'https://weread.qq.com/web/bookListInCategory/{booksId}?maxIndex={page}'.format(booksId=booksId, page=page))
        books = res.json()['books']
        hasMore = res.json()['hasMore']
        print(page,hasMore,books)
        if hasMore == 0: break
        for book in books:
            bookList.append(book)
    return bookList

def writeCSV(bookInfo,path):
    with open(path, 'w', encoding='UTF8', newline='') as f:
        fieldnames = ['title', 'publishTime', 'category', 'intro', 'maxFreeChapter', 'newRating', 'free',
                      'price',
                      'cover']
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval='intro', extrasaction='ignore')

        # 写入头
        writer.writeheader()

        for book in bookInfo:
            # 写入数据
            writer.writerow(book['bookInfo'])

def getWXBookTypeList(BookIds):
    for i in BookIds:
        parentPath = r'C:\Users\Administrator\PycharmProjects\yolov5-dnf\wx-read\{dirName}'.format(dirName=i['title'])
        if not os.path.exists(parentPath):
            os.mkdir(parentPath)
            for t in i['sublist']:
                subPath = r'{parentPath}\{csvName}-{totalCount}.csv'.format(
                    parentPath=parentPath, csvName=t['title'], totalCount=t['totalCount'])
                hasSubFile = os.path.exists(subPath)
                if not hasSubFile:
                    bookInfo = getWXBooks(t['CategoryId'])
                    writeCSV(bookInfo, subPath)
def thread_pool(sub_f,list):
    with ThreadPoolExecutor(max_workers=4) as executor:
        res = [executor.submit(sub_f, j) for j in list]

homeStore = getHomeStoreModulDict()
BookIds = homeStore['categories'][0:2]
print([i['title'] for i in BookIds])
isPool = True
start = time.time()
if isPool:
    print('多线程')
    thread_pool(getWXBookTypeList,BookIds)
    # pool = ProcessPoolExecutor(max_workers=5)
    # results = pool.map(getWXBookTypeList,BookIds)
else:
    getWXBookTypeList(BookIds)
end = time.time()
print(f'多线程，耗时：{end - start}')


