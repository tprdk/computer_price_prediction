import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn import ensemble
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA

DATA_PATH = "house_data_with_label.csv"
df = pd.read_csv(DATA_PATH)

print(df.head())

# Verideki missing value değerlerini silebilmek için önce NaN yapıyoruz.
df = df.replace(['Belirtilmemiş'], np.nan)

# Veri kümesinin özelliklerinin daha anlamlı olması için kategorize edilmelidir.
cols = df.select_dtypes('object').columns
df[cols] = df[cols].stack().astype('category').cat.codes.unstack()

# Missing value değerlerini sütun ortalaması ile dolduruyoruz.
df = df.fillna(df.mean())

# Mean-variance based outlier analizi yapıyoruz
def reject_outliers(data):
    u = np.mean(data["Fiyat"])
    s = np.std(data["Fiyat"])
    data_filtered = data[(data["Fiyat"]>(u-2*s)) & (data["Fiyat"]<(u+2*s))]
    return data_filtered

df=reject_outliers(df)

# Özellikleri ve etiketleri ayırıyoruz
labels = df['Fiyat']
features = df.drop(['İlanNo', 'Fiyat'], axis=1)

# Eğitim ve test kümelerine bölüyoruz
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=2)

# Normalizasyon işlemi olarak StandartScaler ve MinMaxScaler tanımlıyoruz
from sklearn.preprocessing import StandardScaler, MinMaxScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

'''minmax_X = MinMaxScaler()
X_train = minmax_X.fit_transform(X_train)
X_test = minmax_X.transform(X_test)'''
# StandartScaler'ın yarattığı farkı grafiklerle görüyoruz
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.style.use('ggplot')
np.random.seed(1)

scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=df.columns)
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
ax1.set_title('Normalizasyondan Önce')
sns.kdeplot(df['OdaSayısı'], ax=ax1)
sns.kdeplot(df['Isıtma'], ax=ax1)
sns.kdeplot(df['Eşyalı'], ax=ax1)
ax2.set_title('Normalizasyondan Sonra')
sns.kdeplot(scaled_df['OdaSayısı'], ax=ax2)
sns.kdeplot(scaled_df['Isıtma'], ax=ax2)
sns.kdeplot(scaled_df['Eşyalı'], ax=ax2)
plt.show()

'''pipe = Pipeline([
                 ("reduce_dims", PCA()),
                 ("linear", LinearRegression())
                ])


param_grid = dict(reduce_dims__n_components = [0.5, 0.75, 0.95])

grid = GridSearchCV(pipe, param_grid=param_grid, cv=10)
grid.fit(X_train, y_train)
print(grid.score(X_test, y_test))
print(grid.best_params_)'''

# Ilk tahminleyicimizi tanımlıyoruz
regressor = LinearRegression()
regressor = TransformedTargetRegressor(
        regressor=regressor,
        func=np.log, inverse_func=np.exp)
regressor.fit(X_train, y_train) #predicting the test set results
y_pred = regressor.predict(X_test)
print(f'without pca :{regressor.score(X_test, y_test)}')


df1 = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred.flatten()})
df1.plot(kind='bar')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# PCA yöntemi ile bazı özelliklerden kurtuluyoruz

'''pca = PCA()
X_train_new = pca.fit_transform(X_train)
X_test_new = pca.transform(X_test)
print(pca.explained_variance_ratio_)
print(np.argsort(pca.explained_variance_ratio_))

pca = PCA(n_components=1)
X_train_new = pca.fit_transform(X_train)
X_test_new = pca.transform(X_test)

regressor.fit(X_train_new, y_train) #predicting the test set results
y_pred = regressor.predict(X_test_new)
print(f'1 component pca :{regressor.score(X_test_new, y_test)}')

pca = PCA(n_components=6)
X_train_new = pca.fit_transform(X_train)
X_test_new = pca.transform(X_test)

regressor.fit(X_train_new, y_train) #predicting the test set results
y_pred = regressor.predict(X_test_new)
print(f'6 component pca :{regressor.score(X_test_new, y_test)}')

pca = PCA(n_components=7)
X_train_new = pca.fit_transform(X_train)
X_test_new = pca.transform(X_test)

regressor.fit(X_train_new, y_train) #predicting the test set results
y_pred = regressor.predict(X_test_new)
print(f'7 component pca :{regressor.score(X_test_new, y_test)}')

pca = PCA(n_components=5)
X_train_new = pca.fit_transform(X_train)
X_test_new = pca.transform(X_test)

regressor.fit(X_train_new, y_train) #predicting the test set results
y_pred = regressor.predict(X_test_new)
print(f'5 component pca :{regressor.score(X_test_new, y_test)}')

df1 = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred.flatten()})
df1.plot(kind='bar')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
'''

# RFE yöntemi ile bazı özelliklerden kurtuluyoruz
regressor = LinearRegression()
rfe = RFE(regressor, n_features_to_select=10)
rfe = rfe.fit(X_train, y_train)

list(zip(features.columns,rfe.support_,rfe.ranking_))

# predict prices of X_test
y_pred = rfe.predict(X_test)

# evaluate the model on test set
r2 = sklearn.metrics.r2_score(y_test, y_pred)
print(r2)

lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, n_features_to_select=6)
rfe = rfe.fit(X_train, y_train)

# predict prices of X_test
y_pred = rfe.predict(X_test)
r2 = sklearn.metrics.r2_score(y_test, y_pred)
print(r2)

# cross validate ediyoruz
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)
scores = cross_val_score(rfe, X_train, y_train, scoring='r2', cv=folds)


# step-1: create a cross-validation scheme
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)

# step-2: specify range of hyperparameters to tune
hyper_params = [{'n_features_to_select': list(range(1, 16))}]


# step-3: perform grid search
# 3.1 specify model
lm = LinearRegression()
lm.fit(X_train, y_train)
rfe = RFE(lm)

# 3.2 call GridSearchCV()
model_cv = GridSearchCV(estimator = rfe,
                        param_grid = hyper_params,
                        scoring= 'r2',
                        cv = folds,
                        verbose = 1,
                        return_train_score=True)

# fit the model
model_cv.fit(X_train, y_train)

cv_results = pd.DataFrame(model_cv.cv_results_)
print(cv_results)

# plotting cv results
plt.figure(figsize=(16,6))

plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
plt.xlabel('number of features')
plt.ylabel('r-squared')
plt.title("Optimal Number of Features")
plt.legend(['test score', 'train score'], loc='upper left')

# final model
n_features_optimal = 10

lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, n_features_to_select=n_features_optimal)
rfe = rfe.fit(X_train, y_train)

# predict prices of X_test
y_pred = lm.predict(X_test)
r2 = sklearn.metrics.r2_score(y_test, y_pred)
print(r2)


