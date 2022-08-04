import requests
import csv

# 书名
# 出版时间
# 介绍
# 图片
# 推荐值
# 具体分类
res = requests.get('https://weread.qq.com/web/bookListInCategory/700005?maxIndex=99')
books = res.json()['books']

header = ['书名', '出版时间']
with open('wxbooks.csv', 'w', encoding='UTF8',newline='') as f:
    fieldnames = ['title', 'publishTime']
    writer = csv.DictWriter(f,fieldnames=fieldnames,restval='intro',extrasaction='ignore')

    # 写入头
    writer.writeheader()

    for i in books:
        print(i['bookInfo'])
        # 写入数据
        writer.writerow(i['bookInfo'])




