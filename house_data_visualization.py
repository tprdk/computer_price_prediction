import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Veri kümesinin özelliklerinin daha anlamlı olması için kategorize edilmelidir.
'''cols = df.select_dtypes('object').columns
df[cols] = df[cols].stack().astype('category').cat.codes.unstack()
'''

print(df)
print(df.describe())

'''df['OdaSayısı'].value_counts().sort_index().plot(kind='bar')
plt.title('Oda Sayısına göre İlan Dağılımı')
plt.xlabel('OdaSayısı')
plt.ylabel('Miktar')
plt.tight_layout()
plt.show()

df['m²(Net)'].value_counts().sort_index().plot(kind='bar')
plt.title('m²(Net)\'e göre İlan Dağılımı')
plt.xlabel('m² (Net)')
plt.ylabel('Miktar')
plt.tight_layout()
plt.show()

df['Fiyat'].value_counts().sort_index().plot(kind='bar')
plt.title('Fiyata göre İlan Dağılımı')
plt.xlabel('Fiyat')
plt.ylabel('Miktar')
plt.tight_layout()
plt.show()


df['Eşyalı'].value_counts().sort_index().plot(kind='bar')
plt.title('Eşyalılık Durumuna göre İlan Dağılımı')
plt.xlabel('Eşyalılık Durumu')
plt.ylabel('Miktar')
plt.tight_layout()
plt.show()'''


df['Lokasyon'] = [str(semt).split("/")[1] for semt in df['Lokasyon'].values]
print(df['Lokasyon'])

'''df['Lokasyon'].value_counts().sort_index().plot(kind='bar')
plt.title('Semtlere Göre İlan Dağılımı')
plt.xlabel('Semtler')
plt.ylabel('Miktar')
plt.tight_layout()
plt.show()'''


'''df.groupby('Lokasyon')['Fiyat'].mean().plot(kind = "bar")
plt.title('Semtlere Göre Ortalama Kira Fiyatları')
plt.xlabel('Semtler')
plt.ylabel('Fiyat')
plt.tight_layout()
plt.show()

df.groupby('m²(Net)')['Fiyat'].mean().plot(kind = "bar")
plt.title('m²(Net)\'e Göre Ortalama Kira Fiyatları')
plt.xlabel('m²(Net)')
plt.ylabel('Fiyat')
plt.tight_layout()
plt.show()

df.groupby('Isıtma')['Fiyat'].mean().plot(kind = "bar")
plt.title('Isıtma Bilgisine Göre Ortalama Kira Fiyatları')
plt.xlabel('Isıtma Bilgisi')
plt.ylabel('Fiyat')
plt.tight_layout()
plt.show()
'''

