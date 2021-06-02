import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from visualize import visualize_accuracy_graph
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor


def predictor_with_cross_val(X, y, reg_model, model_name, normalization, componenent, count, kfold=10, plot=False):
    try:
        y_pred = cross_val_predict(reg_model, X, y, cv=kfold)
        scores = cross_val_score(reg_model, X, y, cv=kfold)
        if plot:
            start_index = int(np.argmax(scores) * len(y) / kfold)
            end_index = int((np.argmax(scores) + 1) * len(y) / kfold)
            visualize_accuracy_graph(y[start_index:end_index], y_pred[start_index:end_index], model_name, normalization, componenent, count, max(scores))
        return max(scores)
    except Exception as e:
        print(e)
        return 0.0

def create_model(model):
    if model == 'LinearRegression':
        return LinearRegression()
    elif model == 'Lasso' :
        return Lasso()
    elif model == 'Ridge':
        return Ridge()
    elif model == 'GradientBoostingRegressor':
        return GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_split=2,
                                         learning_rate=0.1, loss='ls')
    else:
        return DecisionTreeRegressor()