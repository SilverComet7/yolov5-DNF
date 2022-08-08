import re
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
print(leftDict)