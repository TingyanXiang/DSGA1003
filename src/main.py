#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:16:45 2018

@author: daoyangshan
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from cleaner import get_train_features, get_test_features
from TDE import TimeDependentEnsembler
    
#hyperparam = {"class_weight": "balanced"}
hyperparam = argparse.ArgumentParser(description = "TDE Demo")
# Model-side params
hyperparam.add_argument("--classifier_name", type = str, default = "random_forest", 
                        help = "classifier used in TDE")
hyperparam.add_argument("--n_estimators", type = int, default = 10, 
                        help = "num of estimators in random forest")
hyperparam.add_argument("--criterion", type = str, default = "gini", 
                        help = "purity measurement")
hyperparam.add_argument("--max_depth", type = int, default = None, 
                        help = "maximum depth in tree learners")
hyperparam.add_argument("--max_depth_xgb", type = int, default = 3, 
                        help = "maximum depth in tree learners in XGBoost")
hyperparam.add_argument("--min_samples_split", type = float, default = 2.0, 
                        help = "min num of samples in a node that can be splitted")
hyperparam.add_argument("--class_weight", type = str, default = "balanced", 
                        help = "class weight in tree learners")
hyperparam.add_argument("--scale_pos_weight", type = float, default = 100.0, 
                        help = "class weight adjustment in XGBoost")
hyperparam.add_argument("--learning_rate", type = float, default = 0.1, 
                        help = "learning rate (eta) in XGBoost")

# System-side params
hyperparam.add_argument("--batch_len", type = int, default = 100000, 
                        help = "num of points in a batch")
hyperparam.add_argument("--batches_as_train", type = int, default = 10, 
                        help = "num of batches to train a model")
hyperparam.add_argument("--begin", type = int, default = 0, 
                        help = "begin point (index) of data streaming, 0 indicates from start")
hyperparam.add_argument("--dependency_length", type = int, default = 5, 
                        help = "length of dependency chain")
hyperparam.add_argument("--dependency_weight_decay", type = float, default = 0.8, 
                        help = "weight decay in dependency chain")
hyperparam.add_argument("--phases", type = int, default  = 10,
                        help = "num of ensembling/predicting phases we want to observe")
hyperparam.add_argument("--verbose", type = int, default = 0, 
                        help = "print only final auc (0) or auc for each model in ensembling process (1)")

# Convert argparser to dict
hyperparam = vars(hyperparam.parse_args())
hyperparam["min_samples_split"] = int(hyperparam["min_samples_split"]) if hyperparam["min_samples_split"] >= 1.0 else hyperparam["min_samples_split"]
categorical_features = ['app', 'device', 'os', 'channel']
rare_category_threshold = [50, 50, 50, 50]
TDE = TimeDependentEnsembler(classifier_name = hyperparam["classifier_name"],
                             dependency_length = hyperparam["dependency_length"], 
                             dependency_weight_decay = hyperparam["dependency_weight_decay"], 
                             hyperparam = hyperparam, 
                             train_feature_engineering_function = get_train_features,
                             test_feature_engineering_function = get_test_features, 
                             categorical_features = categorical_features, 
                             rare_category_threshold = rare_category_threshold)
auc_record = []
batch_len = hyperparam["batch_len"]
batches_as_train = hyperparam["batches_as_train"]
begin = hyperparam["begin"]
if hyperparam["verbose"] == 1:
    print ("Verbose mode activated, auc scores in the front reflect the performance of models trained by data near cur batch.")

for i in range(hyperparam["phases"]):
    cur_data = pd.read_csv('/Users/daoyangshan/DSGA1003/train_tail.csv', skiprows = range(1, begin + i * batch_len), 
                           nrows = batch_len * (batches_as_train + 1))
    cur_y = cur_data['is_attributed']
    cur_X = cur_data.drop(['attributed_time', 'is_attributed'], axis = 1)
    X_train, X_test = cur_X.iloc[:batch_len * batches_as_train + 1], cur_X.iloc[batch_len * batches_as_train + 1:]
    y_train, y_test = cur_y.iloc[:batch_len * batches_as_train + 1], cur_y.iloc[batch_len * batches_as_train + 1:]
    TDE.add_new_model(X_train, y_train)
    cur_pred, candidate_preds = TDE.pred_proba(X_test)
    fpr, tpr, _ = metrics.roc_curve(y_test, cur_pred[:, 1])
    auc = metrics.auc(fpr, tpr)
    if hyperparam["verbose"] == 0:
        print ("TDE training, phase " + str(i + 1) + " auc: " + str(auc))
    else:
        candidate_aucs = []
        for pred in candidate_preds:
            fpr, tpr, _ = metrics.roc_curve(y_test, pred[:, 1])
            candidate_aucs.append(metrics.auc(fpr, tpr))
        print ("TDE verbose mode, phase " + str(i + 1) + " auc for all candidate models: " + str(candidate_aucs))
    auc_record.append(auc)
   
print (auc_record)
plt.figure()
plt.plot(auc_record)
#plt.savefig(<dir name>)
#plt.show()