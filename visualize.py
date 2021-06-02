import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams.update({'font.size': 18})

def visualize_distribution(df, col_name, title, x_label, y_label):
    '''
    Veri seti dağılımlarının bastırılma fonksiyonu
    :param df: veri seti
    :param col_name: veri seti kolonları
    :param title: Grafiğin başlığı
    :param x_label: x ekseni başlığı
    :param y_label: y ekseni başlığı
    '''
    df[col_name].value_counts().sort_index().plot(kind='bar')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.show()

def visualize_x_mean_according_to_y(df, x, y, title, x_label, y_label):
    '''
    x özelliğinin y özelliğine göre değişiminin gösterilme fonksiyonu
    :param df: veri seti
    :param x: x özelliği
    :param y: y özelliği
    :param title: Grafiğin başlığı
    :param x_label: x ekseni başlığı
    :param y_label: y ekseni başlığı
    :return:
    '''
    df.groupby(x)[y].mean().plot(kind="bar")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.show()


def visualize_normalization(df, scaled_df, cols):
    '''
    Normalizasyon öncesi ve sonrası farkların gösterilme grafiği
    :param df: Normalizasyon öncesi veri seti
    :param scaled_df: Normalizasyon uygulanmış veri seti
    :param cols: Veri seti kolonları
    '''
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
    '''
    Eğitilen modelin tahminlerinin gösterilme grafiği
    :param y_test: gerçek hedef değerleri
    :param y_pred: tahmin edilen hedef değerleri
    :param model_name: kullanılan model adı
    :param normalization: kullanılan normalizasyon adı
    :param component: kullanılan komponenet adı
    :param count: komponenent analizi için kullanılan n değeri
    :param score: test kümesi üzerinden elde edilen sonuç
    '''
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