import numpy as np
import requests
import pandas as pd
from bs4 import BeautifulSoup

pd.set_option('display.max_columns', None)
cols = ['İşletim Sistemi', 'İşlemci', 'Bellek Kapasitesi', 'Ekran Boyutu (inç)',
        'Ekran', 'Arka kamera çözünürlüğü', 'Ön Kamera Çözünürlüğü', 'Yüz Tanıma',
        'Üretici Garantisi', 'Fiyat']

url_mediamarkt = 'https://www.mediamarkt.com.tr/tr/category/_cep-telefonlar%C4%B1-504171.html?searchParams=&sort=&view=&page='
phone_url = 'https://www.mediamarkt.com.tr'
phone_links = []

for i in range(1, 14):
    print(f'page : {i}')
    req = requests.get(url_mediamarkt + str(i))
    soup = BeautifulSoup(req.text, "html5lib")
    href_links = soup.find_all('a', {'data-gtm-event': 'EEC_PRODUCT_CLICK'}, href=True)
    for item in href_links:
        phone_links.append(phone_url + item['href'])

df = pd.DataFrame(columns=cols, index=np.arange(0, len(phone_links)))
for index, phone in enumerate(phone_links):
    print(f'phone : {index}')
    try:
        req = requests.get(phone)
        soup = BeautifulSoup(req.text, "html5lib")
        dt = soup.find('div', {'class': 'features-list multisection'}).find_all('dt')
        dd = soup.find('div', {'class': 'features-list multisection'}).find_all('dd')
        price = soup.find('div', {'class': 'price big'})
        df.iloc[index, df.columns.get_loc('Fiyat')] = int(price.text.replace(',', '').replace('-', ''))

        for t, d in zip(dt, dd):
            tag = t.text.replace(':', '')
            if tag == 'Arka Kamera Özellikleri' or tag == 'Arka kamera':
                tag = 'Arka kamera çözünürlüğü'
            if tag in cols:
                df.iloc[index, df.columns.get_loc(tag)] = (d.text.split(' ')[0]).replace(',', '').replace('\'', '')
    except:
        print(f'{index}.th phone data couldnt resolve')

print(df.head(10))
df.to_csv('phone_data_with_label.csv')

# get phone prices
# price big length-6
