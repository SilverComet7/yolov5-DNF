import re

from bs4  import BeautifulStoneSoup
import requests

res = requests.get('https://weread.qq.com/web/category/')
soup = BeautifulStoneSoup(res.text)

leftLinks = soup.find_all(class_= re.compile('ranking_list_item_link'))
leftLinks4 = leftLinks[5:]
leftDict = {}
for i in leftLinks4:
    text = i.text
    bookId = i['href'][len('/web/category/'):]
    leftDict[text] = {'全部':bookId}
    # res2 = requests.get('https://weread.qq.com/web/category/subId'.format(subId=bookId))
    # print(res2.text)
    # soup2 = BeautifulStoneSoup(res2.text)
    # subLinks = soup2.find_all(class_=re.compile('ranking_page_header_categroy_item'))
    # print(subLinks)
    bookType = True
    nextBookId = bookId + 1
    while bookType:
        res2 = requests.get('https://weread.qq.com/web/category/subId'.format(subId=nextBookId))
        print(res2.text)
        soup2 = BeautifulStoneSoup(res2.text)
        subLinks = soup2.find_all(class_=re.compile('ranking_page_header_categroy_item'))
        if res2.text & 激活的不是最后一个:
            nextBookId = nextBookId+1
print(leftDict)


# subLinks = soup.find_all(class_= re.compile('ranking_page_header_categroy_item'))
# print(subLinks)
