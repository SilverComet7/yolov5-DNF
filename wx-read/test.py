import requests

res = requests.get('https://weread.qq.com/web/category/100009')
print(res.text)