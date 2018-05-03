#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 14:25:46 2018

@author: kairuo.zhou
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, date, timedelta
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from sklearn import metrics

def convert_str2time(x):
    return time.strptime(x, "%Y-%m-%d %H:%M:%S")

def eda_is_attributed_line_hourly(data):
    stru_time = {'year': 0, 'mon': 1, 'day': 2, 'hour': 3, 'min': 4, 'wday': 6, 'yday': 7 } 
    # categorical variables used for the click_time attribute
    categ_time = ['hour', 'wday']
    # categorical variables
    ff = data.drop(['attributed_time', 'ip', 'channel', 'app', 'os', 'device'], axis=1)
    ff['time_stru'] = ff['click_time'].apply(convert_str2time)
    for col in categ_time:
        ff[col] = ff['time_stru'].apply(lambda x: x[stru_time[col]])
    ff = ff.drop(['time_stru', 'click_time'], axis=1)
    c = ff.groupby(['hour', 'is_attributed'])['is_attributed'].count()
    holder = np.zeros(24)
    plt.figure(figsize=(10,10))
    for i in range(24):
        if len(c[i]) > 1:
            holder[i] = c[i][1]/c[i][0]
    plt.plot(np.arange(24)+1,holder)
    plt.xlabel('th hour')
    plt.ylabel('rates of attribution')
    plt.title('hourly rates of attribution')
    plt.savefig('eda_is_attributed_line_hourly.png')
    
def eda_category_hist(data,category,num_to_show = 10):
    l = ['attributed_time','click_time','channel','ip','device','os', 'app']
    l.remove(category)
    ff = data.drop(l, axis=1)
    c = ff.groupby([category]).count().sort_values(by = 'is_attributed', ascending=False)
    c[:num_to_show].plot(kind='bar', legend=False)
    plt.title('Most frequent ' + category )
    plt.savefig('Most frequent eda_' + category + '_hist.png')
    c[-num_to_show:].plot(kind='bar', legend=False)
    plt.title('least frequent ' + category)
    plt.savefig('Least frequent eda_' + category + '_hist.png')

def main():
    data = pd.read_csv('train.csv')
    # this dictionary is for convenience when using attrbutes in structrue_time objects 
    stru_time = {'year': 0, 'mon': 1, 'day': 2, 'hour': 3, 'min': 4, 'wday': 6, 'yday': 7 } 
    # categorical variables used for the click_time attribute
    categ_time = ['hour', 'wday']
    # categorical variables
    categ = ['app', 'device', 'os', 'channel']
    eda_is_attributed_line_hourly(data)
    eda_category_hist(data,'app')
    eda_category_hist(data,'ip')
    eda_category_hist(data,'device')
    eda_category_hist(data,'os')
    eda_category_hist(data,'channel')
    

if __name__ == "__main__":
    main()
