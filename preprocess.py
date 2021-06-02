import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE


def feature_correction(df):
    # Sadece semt kısmı bilgisini saklıyoruz ve ilan no kolonunu siliyoruz.
    df['Lokasyon'] = [str(semt).split("/")[1] for semt in df['Lokasyon'].values]
    df = df.drop(['İlanNo'], axis=1)
    return df

def missing_value_analysis(df):
    # Boş verilerin yüzdesini bastırıyoruz.
    df = df.replace(['Belirtilmemiş'], np.nan)

    # yüzde olarak 40'tan yüksek oranda missing value içeren kolonları siliyoruz.
    df = df.drop(['Aidat(TL)', 'Depozito(TL)'], axis=1)

    # Veri kümesinin int ve float olmayan özelliklerinin daha anlamlı olması için kategorize edilmelidir.
    cols = df.select_dtypes('object').columns
    df[cols] = df[cols].stack().astype('category').cat.codes.unstack()

    # Missing value değerlerini sütun modu ile dolduruyoruz.
    df['Eşyalı'] = df['Eşyalı'].fillna(df['Eşyalı'].mode()[0])
    return df

def reject_outliers(data):
    '''
    Veri standart sapma ve ortalama etrafında outlier analizi edilir
    :param data: Veri seti
    :return: outlier verileri çıkarılmış veri
    '''
    u = np.mean(data["Fiyat"])
    s = np.std(data["Fiyat"])
    data_filtered = data[(data["Fiyat"]>(u-2*s)) & (data["Fiyat"]<(u+2*s))]
    return data_filtered


def preprocess_data(df):
    '''
    Preprocess fonksiyonu
    :param df: Pandas dataframe veri seti
    :return: processed dataframe
    '''
    df = feature_correction(df)
    df = missing_value_analysis(df)
    df = reject_outliers(df)
    return df


def normalization_method(X_train, X_test=None, scaler_type='StandartScaler'):
    '''
    Normalizasyon fonksiyonu
    kolon ortalaması 0, standart sapma 1 olacak şekilde normalize ediyoruz
    :param X_train: train veri seti
    :param X_test: test veri seti
    :param scaler_type: str tipinde normalizasyon için kullanılacak yöntem adı
    :return:
    '''
    if scaler_type == 'StandartScaler':
        scaler = StandardScaler()
    elif scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler()
    else:
        return X_train, X_test

    if X_test is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test
    else:
        scaled_df = scaler.fit_transform(X_train)
        return pd.DataFrame(scaled_df, columns=X_train.columns)


def component_analysis(component, X_train, X_test, model, n_features_to_select):
    '''
    Component analizi fonksiyonu
    :param component: str tipinde component analizinin adı
    :param X_train: train veri seti
    :param X_test: test veri seti
    :param model: kullanılacak model
    :param n_features_to_select: komponenet analizinde alınancak n sayısı
    :return: train, test veri setleri ve model
    '''
    if component == 'PCA':
        pca = PCA(n_components=n_features_to_select)
        return pca.fit_transform(X_train), pca.transform(X_test), model
    elif component == 'RFE':
        return X_train, X_test, RFE(estimator=model, n_features_to_select=n_features_to_select)
    else:
        return X_train, X_test, model


def split_features_labels(df):
    '''
    Pandas dataframe içerinde gelen veri setinin özellikleri ve etiketlerinin ayrıldığı fonksiyon
    Sonrasında train ve test olarak veri seti ayrılır
    :param df: pandas dataframe
    :return: X_train, y_train, X_test, y_test
    '''
    labels = df['Fiyat']
    features = df.drop(['Fiyat'], axis=1)
    return train_test_split(features, labels, test_size=0.30, random_state=1)
