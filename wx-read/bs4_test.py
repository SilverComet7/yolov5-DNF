import re
<<<<<<< HEAD

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
        # if res2.text & 激活的不是最后一个:
        #     nextBookId = nextBookId+1
print(leftDict)


# subLinks = soup.find_all(class_= re.compile('ranking_page_header_categroy_item'))
# print(subLinks)
=======
from bs4 import BeautifulSoup
import requests
from selenium_test import seleniumOpen
res = requests.get('https://weread.qq.com/web/category/')
soup = BeautifulSoup(res.text)

leftLinks = soup.find_all(class_=re.compile('ranking_list_item_link'))
leftLinks5 = [leftLinks[5]]

leftDict = {

}
for i in leftLinks5:
    text = i.text
    bookId = i['href'][len('/web/category/'):]
    leftDict[text] = {'全部':bookId}
    nextPage = True
    nextBookId = int(bookId) + 1

    while nextPage:
        print('https://weread.qq.com/web/category/{subId}'.format(subId=nextBookId))

        res2 = requests.get('https://weread.qq.com/web/category/subId'.format(subId=nextBookId))

        if res.text == '':
            continue
        backDict  = seleniumOpen(leftDict[text],nextBookId)
        if nextPage:
            nextBookId += 1
        else:
            nextPage  = False
        # soup2 = BeautifulSoup(res2.text,features='html.parser')
        # subLinks = soup2.find_all(class_=re.compile('ranking_page_header_categroy_item'))
        # print(subLinks)
        #
        # if 'active' not in subLinks[-1]:
        #     nextBookId = nextBookId+1
        # else:
        #     bookType = False
>>>>>>> 90a0a541b5bc46641e9550e0f2839c4dc5f87793
