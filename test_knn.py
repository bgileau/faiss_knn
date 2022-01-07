import os
import numpy as np
import pandas as pd
import time

import faiss_knn
import faiss

from sklearn.model_selection import train_test_split
import sklearn.metrics

# test dataset (multiclass):
# https://archive.ics.uci.edu/ml/datasets/Travel+Reviews

## multiclass test case
# data = pd.read_csv(r"C:\Users\brett\Documents\Github\faiss_knn\test_case\letter-recognition.data")

# features = np.array(data.iloc[:,1:])
# labels = np.array(data.iloc[:,0])
# labels = labels.reshape(labels.shape[0],1)

# X_train, X_test, y_train, y_test = train_test_split(features,labels,test_size=.25, random_state=1)
## end multiclass test case

# Binary test case
os_dir = os.path.abspath('')
os_dir_data = os.path.join(os_dir, r"test_case/")

test_csv_path = os_dir_data + "1000_test.csv"
train_csv_path = os_dir_data + "1000_train.csv"
header_list = ["# label","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14","f15","f16","f17","f18","f19","f20","f21","f22","f23","f24","f25","f26"]

test_df = pd.read_csv(test_csv_path, names=header_list, skiprows=[0])  # skip row 0 because we are creating our own header
train_df = pd.read_csv(train_csv_path, names=header_list, skiprows=[0])  # skip row 0 because we are creating our own header
df_columns = list(test_df.columns)

df_columns_features = df_columns
df_columns_label = df_columns[0]
df_columns_features.pop(0)

test_label_arr = np.array(test_df[df_columns_label])
train_label_arr = np.array(train_df[df_columns_label])

y_train = train_label_arr
X_train = train_df

y_test = test_label_arr
X_test = test_df
# end binary test


clf = faiss_knn.FaissKNNImpl(k = 5, faiss=faiss)
clf.fitModel_GPU(X_train, y_train) #Training the model
y_pred = clf.predict_GPU(X_test)
proba, unique_labels = clf.predict_proba(X_test)

print(proba)

proba = proba[:,1].reshape(proba[:,1].shape[0],1)

auc_score = sklearn.metrics.roc_auc_score(y_true=y_test, y_score=proba)
print(auc_score)