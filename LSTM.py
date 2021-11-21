import os
import pandas as pd
import numpy as np
from tensorflow import keras
import FinanceDataReader as fdr

path = 'C:/Users/Kyeongjun/Desktop/stock'
list_name = 'stock_list.csv'
sample_name = 'sample_submission.csv'

stock_list = pd.read_csv(os.path.join(path,list_name), encoding='cp949')
stock_list['종목코드'] = stock_list['종목코드'].apply(lambda x : str(x).zfill(6))

start_date = '20210104'
end_date = '20211105'
Business_days = pd.DataFrame(pd.date_range(start_date,end_date,freq='B'), columns = ['Date'])

sample_code = stock_list.loc[0,'종목코드']

sample = fdr.DataReader(sample_code, start = start_date, end = end_date)[['Close']].reset_index()
sample = pd.merge(Business_days, sample, how = 'outer')
sample['weekday'] = sample.Date.apply(lambda x : x.weekday())
sample['weeknum'] = sample.Date.apply(lambda x : x.strftime('%V'))
sample.Close = sample.Close.ffill()
sample = pd.pivot_table(data = sample, values = 'Close', columns = 'weekday', index = 'weeknum')

# 이전 실거래 20일 데이터로 이후 5일 예측

input_num = 20
output_num = 5

LSTM = keras.Sequential()
LSTM.add(keras.layers.Input(shape = (input_num,)))
LSTM.add(keras.layers.Dense(10, activation = "relu"))
LSTM.add(keras.layers.Dense(output_num, activation = "relu"))

Adam = keras.optimizers.Adam(learning_rate = 0.001)
LSTM.compile(optimizer = Adam, loss = 'mae', metrics = ['mae'])

X = np.zeros((44-int(input_num/5),input_num))
Y = np.zeros((44-int(input_num/5),output_num))
for i in range(44-int(input_num/5)) :
    Xi = sample.iloc[i:i+int(input_num/5),]
    Xi = np.array(Xi)
    Xi = Xi.reshape(1,input_num)[0]
    Yi = sample.iloc[int(input_num/5):int(input_num/5)+1,]
    Yi = np.array(Yi)[0]
    X[i] = Xi
    Y[i] = Yi
X = pd.DataFrame(X)
Y = pd.DataFrame(Y)

trainX = X.iloc[:-1,]
trainY = Y.iloc[:-1,]
testX = X.iloc[-1,]
testY = Y.iloc[-1,]

LSTM.fit(trainX, trainY, epochs = 1000, batch_size = 1)

LSTM.evaluate(testX, testY)

sample04 = sample.iloc[0:4,]
sample04 = np.array(sample04)
sample04.reshape(1,20)[0]
