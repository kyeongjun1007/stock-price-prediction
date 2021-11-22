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

sample['Date'] = sample['Date'].astype(str)
sample['Date'] = pd.to_datetime(sample['Date'], format = '%Y-%m-%d')
sample['days'] = sample['Date'] - pd.datetime(2021,1,4,0,0)
sample['days'] = sample['days'].astype(str)
sample['days'] = sample.days.str[:-5].astype(int)

model = LinearRegression()
model.fit(sample.Close, sample.days)
