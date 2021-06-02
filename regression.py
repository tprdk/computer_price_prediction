import pandas as pd
import numpy as np
from preprocess import preprocess_data, split_features_labels, normalization_method, component_analysis
from regressors import predictor_with_cross_val, create_model
from itertools import product

pd.set_option('display.max_columns', None)
DATA_PATH = "house_data_with_label.csv"
df = pd.read_csv(DATA_PATH)

#PARAMS
PCA_COUNT = len(df.columns) - 3

df = preprocess_data(df)
X_train, X_test, y_train, y_test = split_features_labels(df)
#visualize_normalization(df, normalization_std(df), list(df.columns)[:-1])
#X_train, X_test = normalization_std(X_train, X_test, scaler_type='StandartScaler')

normalization_methods = ['None', 'StandartScaler', 'MinMaxScaler']
models = ['LinearRegression', 'Lasso', 'Ridge', 'DecisionTree', 'GradientBoostingRegressor']
components = ['None', 'PCA', 'RFE']


for model_name in models:
    model = create_model(model_name)
    for normalization in normalization_methods:
        X_train_, X_test_ = normalization_method(X_train, X_test, scaler_type=normalization)
        for component in components:
            print(f'\nNormalization : {normalization} - Model : {model_name} - Component : {component}')
            if component == 'None':
                print(f'Without component analysis score : '
                      f'{predictor_with_cross_val(np.r_[X_train_, X_test_], np.r_[y_train, y_test], model, model_name, normalization, component, 0, 10, True)}')
            else:
                print(f'Before {component} score : '
                      f'{predictor_with_cross_val(np.r_[X_train_, X_test_], np.r_[y_train, y_test], model, model_name, normalization, component, 0, 10, True)}')
                # component analysis
                for i in range(5, 14):
                    X_train_, X_test_, model_ = component_analysis(component, X_train, X_test, model,
                                                                   n_features_to_select=i)
                    print(f'{component} with {i} component score : '
                          f'{predictor_with_cross_val(np.r_[X_train_, X_test_], np.r_[y_train, y_test], model_, model_name, normalization, component, i, 10, True)}')