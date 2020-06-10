import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
from pandas import Grouper, DataFrame
from pandas.plotting import lag_plot, autocorrelation_plot
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def draw_centralTedency(dataframe):
    fig = plt.figure()
    gs = gridspec.GridSpec(8, 8)
    ax = plt.gca()


    mean = dataframe.mean().values[0]
    median = dataframe.median().values[0]
    mode = dataframe.mode().values[0]
    sns.distplot(dataframe, ax=ax)
    ax.axvline(mean, color='r', linestyle='--')
    ax.axvline(median, color='g', linestyle='-')
    ax.axvline(mode, color='b', linestyle='-')
    plt.legend({'Mean': mean, 'Median': median, 'Mode': mode})
    plt.show()

    #print(np.median(dataframe))
    #print(dataframe.mean().values)
    #sns.distplot(dataframe, kde=True, rug=True);
    #plt.axvline(np.median(dataframe), color='b', linestyle='--')
    #plt.axvline(dataframe.mean().values[0], color='g', linestyle='--')

    #plt.axvline(np.mean(dataframe), color='b', linestyle='--')
    #plt.axvline(dataframe.mode(), color='b', linestyle='--')
    plt.show()

def timeseries_to_supervised(df, n_in, n_out):
   agg = pd.DataFrame()
   for i in range(n_in, 0, -1):
      df_shifted = df.shift(i).copy()
      df_shifted.rename(columns=lambda x: ('%s(t-%d)' % (x, i)), inplace=True)
      agg = pd.concat([agg, df_shifted], axis=1)

   for i in range(0, n_out):
      df_shifted = df.shift(-i).copy()
      if i == 0:
         df_shifted.rename(columns=lambda x: ('%s(t)' % (x)), inplace=True)
      else:
         df_shifted.rename(columns=lambda x: ('%s(t+%d)' % (x, i)), inplace=True)
      agg = pd.concat([agg, df_shifted], axis=1)
   agg.dropna(inplace=True)
   return agg

def preprocess(dataset, n_in, n_out):


    #plot the data to get insights
    #print(data['date'])
    #print(dataset.mean())
    columns = ['TL ISE', 'USD ISE', 'SP', 'DAX', 'FTSE', 'NIKEEI', 'BOVESPA', 'EU', 'EM']
    years = DataFrame()
    for index, row in pd.concat([dataset['date'],dataset['USD ISE']], axis=1).iterrows():
        years.loc[index, row['date'].month] = row['USD ISE']

    years.boxplot()
    plt.show()


    plt.plot(dataset['USD ISE'])
    dataset.hist()
    plt.show()
    ax = plt.gca()
    dataset.plot(kind='line', x=0, y=3, color='blue', ax=ax)
    dataset.plot(kind='scatter', x=0, y=3, color='red', ax=ax)
    plt.show()

    lag_plot(dataset['USD ISE'])
    plt.show()
    autocorrelation_plot(dataset['USD ISE'])
    plt.show()

    del dataset['date']
    for column in columns:
        draw_centralTedency(dataset[column].to_frame())
    #print(dataset)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataframe = scaler.fit_transform(np.reshape(dataset.values, (dataset.shape[0], dataset.shape[1])))
    dataframe = pd.DataFrame(data=dataframe, columns=columns)
    stock = dataframe['USD ISE'].to_frame()
    features = dataframe.copy()
    del features['USD ISE']
    del features['TL ISE']

    superv_stock = timeseries_to_supervised(stock,n_in,n_out)
    superv_features = timeseries_to_supervised(features,n_in,n_out)

    #Dataset  transformed to supervised problem of series (samples) 30 composed of
    # 1 days is of size [samples = the rows of any of our dataframe, 1 (for each day) ,
    # features = the columns of the features dataframe]:
    samples = superv_stock.shape[0]
    features = features.shape[1]
    steps = 1
    return superv_stock, superv_features, [samples, steps, features]






file_name = 'data_akbilgic.xlsx'
data = pd.read_excel(file_name, header=None, skiprows=2)
data.dropna(inplace=True)
data = data.iloc[0:530]
data.columns = ['date', 'TL ISE', 'USD ISE', 'SP', 'DAX', 'FTSE', 'NIKEEI', 'BOVESPA', 'EU', 'EM']
preprocess(data, 1, 1)


