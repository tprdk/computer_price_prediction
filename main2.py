import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "phone_data_with_label_elif.csv"
df = pd.read_csv(DATA_PATH)
cols = df.select_dtypes('object').columns
df[cols] = df[cols].stack().astype('category').cat.codes.unstack()
df.fillna(df.mean(), inplace=True)
print(df)
from sklearn.model_selection import train_test_split
np.random.seed(0)
df_train, df_test = train_test_split(df, train_size = 0.7, test_size = 0.3, random_state = 100)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
categorical_features = ['İşletim Sistemi', 'İşlemci', 'Bellek Kapasitesi', 'Ekran Boyutu (inç)',
                        'Ekran','Çözünürlük (YxG)', 'Arka kamera çözünürlüğü', 'Ön Kamera Çözünürlüğü', 'Yüz Tanıma',
                        'Üretici Garantisi']
df_train[categorical_features] = scaler.fit_transform(df_train[categorical_features])
print(df_train)

y_train = df_train.pop('Fiyat')
X_train = df_train

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

lm = LinearRegression()
lm.fit(X_train,y_train)
rfe = RFE(lm, 10)
rfe = rfe.fit(X_train, y_train)

list(zip(X_train.columns,rfe.support_,rfe.ranking_))

print(X_train.columns[rfe.support_])

X_train_rfe = X_train[X_train.columns[rfe.support_]]
print(X_train_rfe.head())


def build_model(X, y):
    X = sm.add_constant(X)  # Adding the constant
    lm = sm.OLS(y, X).fit()  # fitting the model
    print(lm.summary())  # model summary
    return X


def checkVIF(X):
    vif = pd.DataFrame()
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by="VIF", ascending=False)
    return (vif)


X_train_new = X_train_rfe.drop(["Bellek Kapasitesi"], axis = 1)
X_train_new = X_train_new.drop(["Üretici Garantisi"], axis = 1)

X_train_new = build_model(X_train_new,y_train)


print(checkVIF(X_train_new))

lm = sm.OLS(y_train, X_train_new).fit()
y_train_price = lm.predict(X_train_new)


fig = plt.figure()
sns.displot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading
plt.xlabel('Errors', fontsize = 18)
plt.show()
'''y_tr = np.log1p(df['Fiyat'])
df.drop(['Fiyat'], axis=1, inplace=True)

train_df, cv_df , y_train, y_cv = train_test_split(df, y_tr, test_size=0.1, random_state=42)

print('Train size: {}, CV size: {}, Test size: {}' .format(train_df.shape, cv_df.shape, test.shape))'''