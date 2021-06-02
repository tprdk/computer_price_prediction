import string
import numpy as np
import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import random

pd.set_option('display.max_columns', None)
cols = ['İlanNo', 'Lokasyon', 'm²(Brüt)', 'm²(Net)', 'OdaSayısı',
        'BinaYaşı', 'BulunduğuKat', 'KatSayısı', 'Isıtma', 'BanyoSayısı',
        'Balkon', 'Eşyalı', 'KullanımDurumu', 'Siteİçerisinde', 'Aidat(TL)',
        'Depozito(TL)', 'Kimden', 'Fiyat']

url_sahibinden_pages = 'https://www.sahibinden.com/kiralik-daire/istanbul?pagingOffset='
house_url = 'https://www.sahibinden.com'
#house_links = []

user_agent_list = [
'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1 Safari/605.1.15',
'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0',
'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0',
'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
]

def get_free_proxies():
    url = "https://free-proxy-list.net/"
    # get the HTTP response and construct soup object
    soup = BeautifulSoup(requests.get(url).content, "html.parser")
    proxies = []
    for row in soup.find("table", attrs={"id": "proxylisttable"}).find_all("tr")[1:]:
        tds = row.find_all("td")
        try:
            ip = tds[0].text.strip()
            port = tds[1].text.strip()
            host = f"{ip}:{port}"
            proxies.append(host)
        except IndexError:
            continue
    return proxies
from fake_useragent import UserAgent
ua = UserAgent()
def get_session(proxies):
    headers = {
        'Host': 'www.sahibinden.com',
        'User-Agent': ua.random,
        'Cookie': 'MS1=https://www.sahibinden.com/category/en/real-estate; vid=668; cdid=FareXv19cbCTdery5b49a52a; MDR=20180606; __gfp_64b=bDEm7fs0Wb7A.7Rrxx3Vc8KWiV2tqUPA6HKxPqxMzzD.Q7; __gads=ID=c46feb38656fe808:T=1531553074:S=ALNI_MZsbdGGUmPpzuJMK0RROk--kk0Y9w; _ga=GA1.2.2138815818.1531553081; nwsh=std; showPremiumBanner=false; showCookiePolicy=true; userLastSearchSplashClosed=true; MS1=https://www.sahibinden.com/category/en/real-estate; st=a6abb06a0b0f9430ea7fdebd78bf1a15232062dddb59afb52b771d194a3529a1e30b6ca15b691061108084738973f686da6e51c3e00daf378',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    # construct an HTTP session
    session = requests.Session()
    # choose one random proxy
    proxy = random.choice(proxies)
    session.proxies = {"http://": proxy, "https://": proxy}
    session.headers = headers
    return session

'''for i in range(2, 20):
    print(f'page : {i}')
    s = get_session(get_free_proxies())
    res = s.get("https://www.sahibinden.com/kiralik-daire/istanbul?pagingOffset="+str(i*20))
    while res.status_code == 429:
        print(res.content.decode())
        print(res.headers)
        time.sleep(3600)
        s = get_session(get_free_proxies())
        res = s.get("https://www.sahibinden.com/kiralik-daire/istanbul?pagingOffset="+str(i*20))
    soup = BeautifulSoup(res.text.strip(), "html5lib")
    href_links = soup.find_all('a', {'class': 'classifiedTitle'}, href=True)
    for item in href_links:
        house_links.append(house_url + item['href'])
        print(house_url + item['href'])
'''

def readFile(fileName):
    fileObj = open(fileName, "r")  # opens the file in read mode
    words = fileObj.read().splitlines()  # puts the file into an array
    fileObj.close()
    return words

house_links = readFile("links.txt")
df = pd.DataFrame(columns=cols, index=np.arange(0, len(house_links)))
for index, house in enumerate(house_links):
    print(f'house : {index} link: {house}')
    s = get_session(get_free_proxies())
    try:
        res = s.get(house)
        while res.status_code == 429:
            print(res.content.decode())
            print(res.headers)
            time.sleep(1200)
            s = get_session(get_free_proxies())
            res = s.get(house)
        soup = BeautifulSoup(res.text.strip(), "html5lib")
        feature = soup.find('ul', {'class': 'classifiedInfoList'}).find_all('strong')
        value = soup.find('ul', {'class': 'classifiedInfoList'}).find_all('span')
        price = soup.find('div', {'class': 'classifiedInfo'}).find('h3')
        location = soup.find('div', {'class': 'classifiedInfo'}).find('h2')

        df.iloc[index, df.columns.get_loc('Fiyat')] = int(price.contents[0].translate(str.maketrans('','',string.whitespace)).strip("TL").replace(".",""))
        df.iloc[index, df.columns.get_loc('Lokasyon')] = location.text.translate(str.maketrans('','',string.whitespace))

        for f, v in zip(feature, value):
            tag = f.text.replace(':', '')
            tag = tag.translate(str.maketrans('','', string.whitespace))
            if tag in cols:
                value = str(v.text).translate(str.maketrans('', '', string.whitespace))
                print(f"tag: {tag}  \n value: {value}")
                df.iloc[index, df.columns.get_loc(tag)] = value
                df.to_csv('house_data_with_label2.csv', index=False)
    except Exception as e:
        print(f'{index}.th house data couldnt resolve \n exception : {e}')


print(df.head(100))
df.to_csv('house_data_with_label2.csv', index=False)

# get phone prices
# price big length-6
