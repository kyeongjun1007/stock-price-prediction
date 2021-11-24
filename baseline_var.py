#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import FinanceDataReader as fdr

from sklearn.linear_model import LinearRegression
from tqdm import tqdm


# ## Get Stock List

# In[2]:


path = 'C:/Users/Kyeongjun/Desktop/stock'
list_name = 'stock_list.csv'
sample_name = 'sample_submission.csv'

stock_list = pd.read_csv(os.path.join(path,list_name), encoding='cp949')
stock_list['종목코드'] = stock_list['종목코드'].apply(lambda x : str(x).zfill(6))
stock_list


# ## Get Data & Modeling

# In[3]:


start_date = '20210104'
end_date = '20211105'

start_weekday = pd.to_datetime(start_date).weekday()
max_weeknum = pd.to_datetime(end_date).strftime('%V')
Business_days = pd.DataFrame(pd.date_range(start_date,end_date,freq='B'), columns = ['Date'])

print(f'WEEKDAY of "start_date" : {start_weekday}')
print(f'NUM of WEEKS to "end_date" : {max_weeknum}')
print(f'HOW MANY "Business_days" : {Business_days.shape}', )
display(Business_days.head())


# ## Baseline 모델의 구성 소개 ( Sample )
# 
# - X : (월 ~ 금) * 43주간
# - y : (다음주 월 ~ 금) * 43주간
#     - y_0 : 다음주 월요일
#     - y_1 : 다음주 화요일
#     - y_2 : 다음주 수요일
#     - y_3 : 다음주 목요일
#     - y_4 : 다음주 금요일
# 
# 
# - 이번주 월~금요일의 패턴을 학습해 다음주 월요일 ~ 금요일을 각각 예측하는 모델을 생성
#     
# - 이 과정을 모든 종목(370개)에 적용

# In[4]:


sample_code = stock_list.loc[0,'종목코드']

sample = fdr.DataReader(sample_code, start = start_date, end = end_date)[['Close']].reset_index()
sample = pd.merge(Business_days, sample, how = 'outer')
sample['weekday'] = sample.Date.apply(lambda x : x.weekday())
sample['weeknum'] = sample.Date.apply(lambda x : x.strftime('%V'))
sample.Close = sample.Close.ffill()
sample = pd.pivot_table(data = sample, values = 'Close', columns = 'weekday', index = 'weeknum')
sample.head()

# In[5]:


model = LinearRegression()


# In[6]:


x = sample.iloc[0:-2].to_numpy()
x.shape


# In[7]:


y = sample.iloc[1:-1].to_numpy()
y_0 = y[:,0]
y_1 = y[:,1]
y_2 = y[:,2]
y_3 = y[:,3]
y_4 = y[:,4]

y_values = [y_0, y_1, y_2, y_3, y_4]


# In[8]:


x_public = sample.iloc[-2].to_numpy()


# - 예측

# In[9]:


predictions = []
for y_value in y_values :
    model.fit(x,y_value)
    prediction = model.predict(np.expand_dims(x_public,0))
    predictions.append(prediction[0])
predictions
y_values

# - 실제 Public 값

# In[10]:


sample.iloc[-1,0:5].values


# # 전체 모델링

# In[11]:


sample_name = 'sample_submission.csv'
sample_submission = pd.read_csv(os.path.join(path,sample_name))


# In[12]:


model = LinearRegression()
for code in tqdm(stock_list['종목코드'].values):
    data = fdr.DataReader(code, start = start_date, end = end_date)[['Close']].reset_index()
    data = pd.merge(Business_days, data, how = 'outer')
    data['weekday'] = data.Date.apply(lambda x : x.weekday())
    data['weeknum'] = data.Date.apply(lambda x : x.strftime('%V'))
    data.Close = data.Close.ffill()
    data = pd.pivot_table(data = data, values = 'Close', columns = 'weekday', index = 'weeknum')
 
    x = data.iloc[0:-2].to_numpy() # 2021년 1월 04일 ~ 2021년 10월 22일까지의 데이터로
    y = data.iloc[1:-1].to_numpy() # 2021년 1월 11일 ~ 2021년 10월 29일까지의 데이터를 학습한다.
    y_0 = y[:,0]
    y_1 = y[:,1]
    y_2 = y[:,2]
    y_3 = y[:,3]
    y_4 = y[:,4]

    y_values = [y_0, y_1, y_2, y_3, y_4]
    x_public = data.iloc[-2].to_numpy() # 2021년 11월 1일부터 11월 5일까지의 데이터를 예측할 것이다.
    
    predictions = []
    for y_value in y_values :
        model.fit(x,y_value)
        prediction = model.predict(np.expand_dims(x_public,0))
        predictions.append(prediction[0])
    sample_submission.loc[:,code] = predictions * 2
sample_submission.isna().sum().sum()


# In[13]:


sample_submission.columns


# In[14]:


columns = list(sample_submission.columns[1:])

columns = ['Day'] + [str(x).zfill(6) for x in columns]

sample_submission.columns = columns


# In[15]:


sample_submission.to_csv('BASELINE_Linear3.csv',index=False)


# In[16]:


sample_submission

