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

# https://weread.qq.com/web/bookListInCategory/newbook?maxIndex=20&rank=1

typesDict = {
    'top50飙升':'rising',
    'top50/新书':'newbook',
    'top200/总榜':'all',
    '神作':'newrating_publish',
    '神作/潜力':'newrating_potential_publish',
     "精品小说":{
        
     },
     "历史":{

     },
     "文学":{

     },
     "艺术":{

     },
     "人物传记":{

     },
     "哲学宗教":{

     },
     "计算机":{
      
     }
}

bookTypeIds = {'精品小说': '100000', '历史': '200000', '文学': '300000', '艺术': '400000', '人物传记': '500000', '哲学宗教': '600000', '计算机': '700000', '心理': '800000', '社会文化': '900000', '个人成长': '1000000', '经济理财': '1100000', '政治军事': '1200000', '童书': '1300000', '教育学习': '1400000', '科学技术': '1500000', '生活百科': '1600000', '期刊专栏': '1700000', '原版书': '1800000', '医学健康': '2100000', '男生小说': '1900001', '女生小说': '2000001'}


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



BookId = [700003,700004,700005,700006,700007]
for i in BookId:
    print(i)
    bookInfo = getWXBooks(i)

    with open('{csvName}.csv'.format(csvName=i), 'w', encoding='UTF8', newline='') as f:
        fieldnames = ['title', 'publishTime', 'category', 'intro',  'maxFreeChapter','newRating', 'free', 'price', 'cover']
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval='intro', extrasaction='ignore')

        # 写入头
        writer.writeheader()

        for book in bookInfo:
            # 写入数据
            writer.writerow(book['bookInfo'])
