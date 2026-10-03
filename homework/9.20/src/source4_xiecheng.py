import requests
from bs4 import BeautifulSoup

url = 'https://you.ctrip.com/sight/yinchuan239/1414597.html'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/122.0.0.0 Safari/537.36',
    'Referer': 'https://you.ctrip.com/',
    'Accept-Language': 'zh-CN,zh;q=0.9',
}
resp = requests.get(url, headers=headers, timeout=10)
resp.encoding = 'utf-8'
soup = BeautifulSoup(resp.text, 'lxml')
print(len(resp.text))
divs = soup.find_all('div', class_='commentDetail')
print(len(divs), divs[:1])