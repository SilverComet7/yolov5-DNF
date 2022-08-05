import time
import random
import requests
import csv

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
