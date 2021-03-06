#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 09:04:42 2018

@author: daoyangshan
"""

import numpy as np
import collections
import copy
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

class TimeDependentEnsembler():
    def __init__(self, classifier_name, dependency_length, dependency_weight_decay, hyperparam,
                train_feature_engineering_function, test_feature_engineering_function,
                categorical_features, rare_category_threshold):
        """
        Constructor
        
        classifier_name (String): classifier we use for each model, either "dicision_tree" or "random_forest"
        dependency_length (int): number of models we use in emsembler
        dependency_weight_decay (float): decay rate for the weight of previous models
        hyperparam (dict): hyperparameters for each model
        train_feature_engineering_function (func): function used for feature engineering on train set
        test_feature_engineering_function (func): function used for feature engineering on test set
        model_queue (deque): a queue to store functions used for emsembling
        dict_queue (deque): a queue to store dicts used for feature engineering
        categorical_features (list): feature (names) that should be considered as categorical feature
        rare_category_threshold (list): threshold for a category to be considered rare in the corresponding feature
        """
        supported_classifier = set(["decision_tree", "random_forest", "xgboost", "lightgbm"])
        if classifier_name not in supported_classifier:
            raise ValueError("Unidentified classifier name, see document for supported classifiers")
        self.classifier_name = classifier_name
        self.dependency_length= dependency_length
        self.dependency_weight_decay = dependency_weight_decay
        self.hyperparam = hyperparam
        self.train_feature_engineering_function = train_feature_engineering_function
        self.test_feature_engineering_function = test_feature_engineering_function
        self.model_queue = collections.deque()
        self.dict_queue = collections.deque()
        self.categorical_features = categorical_features
        self.rare_category_threshold = rare_category_threshold
        
    def alter_hyperparam(self, new_hyperparam):
        """
        Change hyperparam used for future classifiers
        """
        self.hyperparam = new_hyperparam
        
    def add_new_model(self, X, y):
        """
        Given X and y as train set, train a new model and push it to the end of queue
        
        X (np array): train features
        y (np array): train labels
        """
        new_classifier = None  
        # Expand the hyperparam block if needed
        if self.classifier_name == "random_forest":
            new_classifier = RandomForestClassifier(n_estimators = self.hyperparam.get("n_estimators"), 
                                                    criterion = self.hyperparam.get("criterion"), 
                                                    max_depth = self.hyperparam.get("max_depth"),
                                                    min_samples_split = self.hyperparam.get("min_samples_split"),
                                                    class_weight = self.hyperparam.get("class_weight"))
        elif self.classifier_name == "decision_tree":
            new_classifier = DecisionTreeClassifier(criterion = self.hyperparam.get("criterion"),
                                                    max_depth = self.hyperparam.get("max_depth"),
                                                    min_samples_split = self.hyperparam.get("min_samples_split"),
                                                    class_weight = self.hyperparam.get("class_weight"))
        elif self.classifier_name == "xgboost":
            new_classifier = XGBClassifier(silent = True, objective = "binary:logistic", 
                                           learning_rate = self.hyperparam.get("learning_rate"),
                                           n_estimators = self.hyperparam.get("n_estimators"),
                                           max_depth = self.hyperparam.get("max_depth_xgb"),
                                           scale_pos_weight = self.hyperparam.get("scale_pos_weight"))
        else:
            # LightGBM
            new_classifier = None
        # Feature engineering on train set
        X, cur_dict = self.train_feature_engineering_function(X, self.categorical_features, 
                                                              self.rare_category_threshold)
        new_classifier.fit(X, y)
        self.model_queue.appendleft(new_classifier)
        self.dict_queue.appendleft(cur_dict)
        if len(self.model_queue) > self.dependency_length:
            self.model_queue.popleft()
            self.dict_queue.popleft()
        return self
            
    def pred_proba(self, X):
        """
        Given test set X (which is not being feature engineered), generate prediction for X
        
        X (np array): test features
        
        return: class probability matrix: (num_instances, 2), all predict proba matrix (list of matrices)
        """
        total_weight = 0
        cur_weight = 1
        # Here we assume it's a binary classification
        total_pred = np.array([[0.0, 0.0] for _ in range(X.shape[0])])
        all_preds = []
        for i in range(len(self.model_queue) - 1, -1, -1):
            X_cur = copy.deepcopy(X)
            X_cur = self.test_feature_engineering_function(X_cur, self.categorical_features, 
                                                           self.rare_category_threshold,
                                                           self.dict_queue[i])
            # Potential optimization here: use 3d matrix to store all preds and reduce at the end
            cur_pred = self.model_queue[i].predict_proba(X_cur)
            total_pred += cur_pred * cur_weight
            all_preds.append(cur_pred)
            total_weight += cur_weight
            cur_weight *= self.dependency_weight_decay
        return total_pred / total_weight, all_preds
    
