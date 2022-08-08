import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager


def seleniumOpen(parentDict,bookId=700000):
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    driver.get('https://weread.qq.com/web/category/{bookId}'.format(bookId=bookId))
    classList = driver.find_elements(By.CLASS_NAME, "ranking_page_header_categroy_item")
    print(classList[1:])
    for i in classList[1:]:
        print(i.text)
        i.click()
        time.sleep(10)
        print(driver.current_url)
    classListContent = driver.find_element(By.CLASS_NAME, "ranking_page_header_categroy_container")
    activeClass = classListContent.find_element(By.CLASS_NAME, 'active')

    parentDict[activeClass.text] = bookId
    return {
       'nextPage':classList[-1].text != activeClass.text,
    }
    # nextPage = True
    # if classList[-1].text != activeClass.text:
    #     print('next sub page')
    # else:
    #     print('is last sub page')
    #     nextPage = False
    # return nextPage

seleniumOpen({})