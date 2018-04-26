#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 09:04:41 2018

@author: daoyangshan
"""

import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder

def convert_str2time(x): 
    '''conver time to the structure form
    output: (year, mon, day, hour, min, sec, wday, yday, isdst)
    '''
    return time.strptime(x, "%Y-%m-%d %H:%M:%S")

def match_ip_freq(freq, df):
    '''
    map ip to ip_counts according to 
    freq (pandas.Series): with index=ip and value=counts
    '''
    def match_ip(x, freq):
        try:
            return freq.loc[x]
        except:
            return 0
    return df.assign(ip_counts = (df['ip'].apply(lambda x: match_ip(x, freq)).values))


def bin_class_train(df, categorical_list, threshold_rare):
    '''
    bin rare classes and add one unknown class (0) in case there are new classes in test
    categorical_list (list): all categorical variables for binning 
    threshold_rare (list): threshold of rare classes for each categorical variable
    '''
    num_classes = []
    rare = []
    com = []
    for i in range(len(categorical_list)):
        col = categorical_list[i]
        threshold = threshold_rare[i]
        categ = df[col]
        categ_count = categ.value_counts()
        rare_class =set(categ_count[categ_count.values<threshold].index.values)
        rare.append(rare_class)
        com_class = list(categ_count[categ_count.values>=threshold].index.values)
        com.append(com_class)
        n_class = len(com_class)
        if len(rare_class) != 0: #rare class is the last class
            n_class += 1
            categ = categ.apply(lambda x: n_class if x in rare_class else com_class.index(x)+1)
        else:
            categ = categ.apply(lambda x: com_class.index(x)+1)
        num_classes.append(n_class+1) #add one class for 'unknown'  
        df[col] = categ.values
    return df, num_classes, rare, com

def bin_class_test(df, categorical_list, threshold_rare, num_classes, rare, com):
    '''
    bin class in test based on training sets
    categorical_list (list): all categorical variables for binning 
    threshold_rare (list): threshold of rare classes for each categorical variable
    num_classes (list): the number of classes for each categorical variable
    rare (list of sets): index of rare classes for each categorical variable
    com (list of lists): index of common classes for each caregorical variable
    '''
    def get_class(x, com_class):
        try:
            return com_class.index(x)+1
        except:
            return 0
    for i in range(len(categorical_list)):
        col = categorical_list[i]
        tsh = threshold_rare[i]
        categ = df[col]
        n_class = num_classes[i]
        rare_class = rare[i]
        com_class = com[i]
        if len(rare_class) != 0:
            categ = categ.apply(lambda x: n_class-1 if x in rare_class else get_class(x, com_class))
        else:
            categ = categ.apply(lambda x: get_class(x, com_class)) 
        df[col] = categ.values
    return df

def get_train_features(df, categorical_list, threshold_rare):
    '''
    get train features
    categorical_list (list): all categorical variables for binning 
    threshold_rare (list): threshold of rare classes for each categorical variable
    '''
    #ip_counts
    freq_ip = df['ip'].value_counts() #Series: from ip to count
    df = match_ip_freq(freq_ip, df) #get ip_counts
    #binning rare classes and one-hot-encoder
    df, num_classes, rare, com = bin_class_train(df, categorical_list, threshold_rare)
    enc = OneHotEncoder(n_values=np.array(num_classes))
    enc.fit(df[categorical_list])
    names = [] #names for dummy variables
    for i, cl in enumerate(enc.n_values_):
        names += [categorical_list[i]+str(j) for j in range(cl)]
    categ_dummy = enc.transform(df[categorical_list]).toarray()
    categ_dummy = pd.DataFrame(categ_dummy, columns=names)

    # deal with click_time
    time_stru = df['click_time'].apply(convert_str2time)
    # categorical variables used for the click_time attribute
    categ_time = ['hour', ]
    stru_dict = {'year': 0, 'mon': 1, 'day': 2, 'hour': 3, 'min': 4, 'wday': 6, 'yday': 7 }
    col = categ_time[0]
    h_arr = time_stru.apply(lambda x: x[stru_dict[col]]).values.reshape(-1,1)
    enc_time = OneHotEncoder(n_values=24)
    enc_time.fit(h_arr)
    h_dummy = enc_time.transform(h_arr).toarray()
    h_dummy = pd.DataFrame(h_dummy, columns=['h_'+str(i) for i in range(24)])
  
    df.reset_index(drop=True, inplace=True)
    df = df.drop(['click_time', 'ip']+categorical_list, axis=1)
    df = pd.concat((df, categ_dummy, h_dummy), axis=1)
    feature_engineer = {'Freq_ip': freq_ip, 'OneHotEncoder_categorical':enc, 
                        'Num_classes': num_classes, 'Names_categorical': names,
                        'RareClass': rare, 'CommonClass': com, 
                        'OneHotEncoder_time': enc_time }
                    
    return df, feature_engineer

def get_test_features(df, categorical_list, threshold_rare, feature_engineer):
    '''
    get test features 
    categorical_list (list): all categorical variables for binning 
    threshold_rare (list): threshold of rare classes for each categorical variable
    feature_engineer (dictionary): information to transform the test dataset 
    '''
    freq_ip = feature_engineer['Freq_ip']
    enc = feature_engineer['OneHotEncoder_categorical']
    num_classes = feature_engineer['Num_classes']
    names = feature_engineer['Names_categorical']
    rare = feature_engineer['RareClass']
    com = feature_engineer['CommonClass']
    enc_time = feature_engineer['OneHotEncoder_time']
    df = match_ip_freq(freq_ip, df)
    df = bin_class_test(df, categorical_list, threshold_rare, num_classes, rare, com)
    categ_dummy = enc.transform(df[categorical_list]).toarray()
    categ_dummy = pd.DataFrame(categ_dummy, columns=names)

    # deal with click_timecfc
    time_stru = df['click_time'].apply(convert_str2time)
    # categorical variables used for the click_time attribute
    categ_time = ['hour', ]
    stru_dict = {'year': 0, 'mon': 1, 'day': 2, 'hour': 3, 'min': 4, 'wday': 6, 'yday': 7 }
    col = categ_time[0]
    h_arr = time_stru.apply(lambda x: x[stru_dict[col]]).values.reshape(-1,1)
    h_dummy = enc_time.transform(h_arr).toarray()
    h_dummy = pd.DataFrame(h_dummy, columns=['h_'+str(i) for i in range(24)])
    
    df.reset_index(drop=True, inplace=True)
    df = df.drop(['click_time', 'ip']+categorical_list, axis=1)
    df = pd.concat((df, categ_dummy, h_dummy), axis=1)
    return df