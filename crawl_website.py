import string
import numpy as np
import requests
import pandas as pd
from bs4 import BeautifulSoup, NavigableString

pd.set_option('display.max_columns', None)
cols = ['İşletim Sistemi', 'İşlemci', 'Bellek Kapasitesi', 'Ekran Boyutu (inç)',
        'Ekran', 'Çözünürlük (YxG)', 'Arka kamera çözünürlüğü', 'Ön Kamera Çözünürlüğü', 'Yüz Tanıma',
        'Üretici Garantisi', 'Fiyat']

url_mediamarkt = 'https://www.mediamarkt.com.tr/tr/category/_cep-telefonlar%C4%B1-504171.html?searchParams=&sort=&view=&page='
phone_url = 'https://www.mediamarkt.com.tr'
phone_links = []

for i in range(1, 14):
    #print(f'page : {i}')
    file = open("phone_links.txt", mode='w', encoding='utf-8')
    req = requests.get(url_mediamarkt + str(i))
    soup = BeautifulSoup(req.text, "html5lib")
    href_links = soup.find_all('a', {'data-gtm-event': 'EEC_PRODUCT_CLICK'}, href=True)
    for item in href_links:
        phone_links.append(phone_url + item['href'])

df = pd.DataFrame(columns=cols, index=np.arange(0, len(phone_links)))
for index, phone in enumerate(phone_links):
    print(f'phone : {index} link: {phone}')
    try:
        req = requests.get(phone)
        soup = BeautifulSoup(req.text, "html5lib")
        dt = soup.find('div', {'class': 'features-list multisection'}).find_all('dt')
        dd = soup.find('div', {'class': 'features-list multisection'}).find_all('dd')
        price = soup.find('div', {'class': 'price big'})
        df.iloc[index, df.columns.get_loc('Fiyat')] = int(price.text.replace(',', '').replace('-', ''))

        for t, d in zip(dt, dd):
            tag = t.text.replace(':', '')
            if (tag == 'Arka Kamera Özellikleri' or tag == 'Arka kamera') and "megapixel" not in str(df.iloc[index, df.columns.get_loc('Arka kamera çözünürlüğü')]):
                tag = 'Arka kamera çözünürlüğü'
            if tag in cols:
                value = ""
                for val in d.contents:
                    if isinstance(val, NavigableString):
                        value += val.split('\n')[0].split("/")[0]
                    else:
                        value += val.text.split('\n')[0].split(",")[0]
                value = str(value).translate(str.maketrans('', '', string.punctuation))
                if len(value.split(" ")) > 1:
                    value = value.split(" ")[0] +" "+ value.split(" ")[1]
                #value = (d.text.split(' ')[0]).replace(',', '').replace('\'', '')
                if tag == 'İşletim Sistemi':
                    value = value.split(" ")[0][ 0 : int(len(value.split(" ")[0]) / 2)]
                    if len(value.split(" ")) > 1:
                        value += value.split(" ")[1]
                print(f"tag: {tag}  \n value: {value}")
                df.iloc[index, df.columns.get_loc(tag)] = value
    except Exception as e:
        print(f'{index}.th phone data couldnt resolve \n exception : {e}')


print(df.head(100))
df.to_csv('phone_data_with_label_elif.csv', index=False)

# get phone prices
# price big length-6
