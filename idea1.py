import pandas as pd
import numpy as np
import os
import FinanceDataReader as fdr
from datetime import date
import datetime
from sklearn.linear_model import LinearRegression


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
sample = sample.dropna()

sample['Date'] = sample['Date'].astype(str)
sample['Date'] = pd.to_datetime(sample['Date'], format = '%Y-%m-%d')
sample['days'] = sample['Date'] - pd.datetime(2021,1,4,0,0)
sample['days'] = sample['days'].astype(str)
sample['days'] = sample.days.str[:-5].astype(int)
sample['Close'][0]
## Close mean
sample['meanC'] = sample['Close'][0]
for i in range(1,len(sample)) :
    sample.iloc[i,3] = np.mean(sample.iloc[0:i+1,1])

## month trend
sample['trend_m'] = 0
for i in range(21,len(sample)) :
    model_m = LinearRegression()
    model_m.fit(np.array(sample.iloc[i-21:i-1,2]).reshape(-1,1), sample.iloc[i-21:i-1,1])
    sample.iloc[i,4] = model_m.coef_

## week trend
sample['trend_w'] = 0
for i in range(6,len(sample)) :
    model_w = LinearRegression()
    model_w.fit(np.array(sample.iloc[i-6:i-1,2]).reshape(-1,1), sample.iloc[i-6:i-1,1])
    sample.iloc[i,5] = model_w.coef_

model = LinearRegression()
model.fit(sample.iloc[21:,3:6].to_numpy(), sample.iloc[21:,1].to_numpy())
model.predict(np.expand_dims(sample.iloc[len(sample)-1,3:6].to_numpy(),0))
sample.iloc[len(sample)-1,1]

sample.Close[209]
sample['Close']
sample.iloc[208,]
