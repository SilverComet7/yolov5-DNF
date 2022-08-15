import random
import time

import pymongo
import redis
import requests

serarchWordList = ['unity', 'vue', '逆向','爬虫','机器学习']
client = redis.StrictRedis()


def mongodbWrite(collectName, bookInfo):
    client = pymongo.MongoClient()
    dataBase = client['booksInfo']
    collect = dataBase[collectName]
    collect.insert_many(bookInfo)


def sortWay(book):
    return book['bookInfo']['newRating']


def getWXBooks(queryPath, searchWord):
    # todo redis
    # booksTitleSet = client.hget(searchKeyWord, 'booksTitleSet') or set()
    # maxIdx = int(client.hget(searchKeyWord, 'maxIdx')) if client.hget(searchKeyWord, 'maxIdx') is not None else 0
    maxIdx = 0
    bookList = []
    hasMore = 1
    totalCount = 0
    while hasMore:
        # maxIdx = i * 20  # example: i = 20 时候出现错误中断了(反爬触发，断电等) 记录到redis 后续继续从 i=20开始
        rTime = random.randint(1, 3)  # 随机延迟，从1到3内取一个整数值
        time.sleep(rTime)  # 把随机取出的整数值传到等待函数中
        res = requests.get(
            queryPath, params={'maxIdx': maxIdx, 'keyword': searchWord, 'fragmentSize': 120, 'count': 20})
        books = res.json()['books']
        if not totalCount:
            totalCount = res.json()['totalCount']
        if len(books):
            for book in books:
                # bookTitle = book['bookInfo']['title']
                print(book)
                # # 书名不在redis中
                # if bookTitle not in client.hget(searchKeyWord, 'booksTitleSet'):
                bookList.append(book)
                # booksTitleSet.add(bookTitle)
        hasMore = res.json()['hasMore']
        if hasMore == 0: break
        maxIdx += 20  # opencv:{maxSpiderIndex:20,bookTitle:{1,2,3}}
        # client.hmset(searchKeyWord, {
        #     'maxIdx': maxIdx,
        #     'booksTitleSet': booksTitleSet
        # })

    # bookList.sort(key=sortWay, reverse=True)
    # writeCSV(bookList, r'globalSearchInfo\{searchKeyWord}-{totalCount}.csv'.format(searchKeyWord=searchKeyWord,
    #                                                                              totalCount=totalCount))
    # mongodb 记录
    return bookList


if __name__ == '__main__':
    for i in serarchWordList:
        books = getWXBooks('https://weread.qq.com/web/search/global', i)
        mongodbWrite(i, books)
