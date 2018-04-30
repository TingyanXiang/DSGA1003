import pandas as pd
import copy
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from cleaner import get_train_features, get_test_features

chunk_len = 300000
auc_record = []
categorical_features = ['app', 'device', 'os', 'channel']
rare_category_threshold = [50, 50, 50, 50]
test_data = pd.read_csv('/Users/daoyangshan/DSGA1003/train_tail.csv', skiprows = range(1, chunk_len * 10 + 1), nrows = chunk_len)
test_X = test_data.drop(['attributed_time', 'is_attributed'], axis = 1)
test_y = test_data['is_attributed']
for i in range(10):
    cur_train = pd.read_csv('/Users/daoyangshan/DSGA1003/train_tail.csv', 
                            skiprows = range(1, chunk_len * i + 1), nrows = chunk_len)
    cur_X = cur_train.drop(['attributed_time', 'is_attributed'], axis = 1)
    cur_y = cur_train['is_attributed']
    clf = DecisionTreeClassifier(class_weight = "balanced")
    cur_X, cur_dict = get_train_features(cur_X, categorical_features, rare_category_threshold)
    clf.fit(cur_X, cur_y)
    
    cur_test = copy.deepcopy(test_X)
    cur_test = get_test_features(cur_test, categorical_features, rare_category_threshold, cur_dict)
    cur_pred = clf.predict_proba(cur_test)
    fpr, tpr, _ = metrics.roc_curve(test_y, cur_pred[:, 1])
    auc = metrics.auc(fpr, tpr)
    print ("cur auc: " + str(auc))
    auc_record.append(auc)
    
print (auc_record)