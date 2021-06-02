import time
from threading import Thread

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams.update({'font.size': 18})

def visualize_distribution(df, col_name, title, x_label, y_label):
    df[col_name].value_counts().sort_index().plot(kind='bar')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.show()

def visualize_x_mean_according_to_y(df, x, y, title, x_label, y_label):
    df.groupby(x)[y].mean().plot(kind="bar")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.show()

def visualize_normalization(df, scaled_df, cols):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 10))
    ax1.set_title('Normalizasyondan Önce')
    ax1.set_xlim(-50, 150)
    ax2.set_title('Normalizasyondan Sonra')
    legends = []
    for col in cols:
        sns.kdeplot(df[col], ax=ax1)
        sns.kdeplot(scaled_df[col], ax=ax2)
        legends.append(col)
    plt.legend(legends)
    plt.show()

def visualize_accuracy_graph(y_test, y_pred, model_name, normalization, component, count, score):
    df1 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
    df1.plot(kind='bar')
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.title(f'Model : {model_name}'
              f' - Normalizasyon : {normalization}'
              f' - Komponent : {component}'
              f'\nBaşarı : {score}')

    fig = plt.gcf()
    fig.set_size_inches((16, 10), forward=False)

    plt.savefig(f'plot/{model_name}/m_{model_name}_n_{normalization}_c_{component}_{count}.png', dpi=500)
    plt.close()
    time.sleep(0.2)