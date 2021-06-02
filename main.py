import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "phone_data_with_label_elif.csv"

categorical_features = ['İşletim Sistemi', 'İşlemci', 'Bellek Kapasitesi', 'Ekran Boyutu (inç)',
                        'Ekran','Çözünürlük (YxG)', 'Arka kamera çözünürlüğü', 'Ön Kamera Çözünürlüğü', 'Yüz Tanıma',
                        'Üretici Garantisi']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

df = pd.read_csv(DATA_PATH)

print(df.head())
features = df.drop(['Fiyat'], axis=1)
labels = df['Fiyat']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=42, shuffle=True)

transformer = ColumnTransformer(
    transformers=[('cat', categorical_transformer, categorical_features)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', transformer),
                      ('classifier', LogisticRegression())])


clf.fit(X_train, y_train)
print("logistic regression model score: %.3f" % clf.score(X_test, y_test))

reg = Pipeline(steps=[('preprocessor', transformer),
                      ('classifier', LinearRegression())])
reg.fit(X_train, y_train)
print("linear regression model score: %.3f" % reg.score(X_test, y_test))

clf = ensemble.GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_split=2,
                                         learning_rate=0.1, loss='ls')
clf = Pipeline(steps=[('preprocessor', transformer),
                      ('classifier', clf)])
clf.fit(X_train, y_train)
print("GradientBoostingRegressor model score: %.3f" % clf.score(X_test, y_test))
