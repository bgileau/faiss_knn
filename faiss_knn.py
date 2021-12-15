#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: bgileau-shankarpm

Changes: 
More robust type handling to make faiss easier to interact with + input sanitization to lessen the initial burden of using faiss.
Introduction of predict_proba functionality, allowing greater compatibility with external packages (i.e. sklearn). Supports binary and multi-class.
    With predict_proba, calculation of AUC (among others) is very easy. However, to avoid adding a dependency here, it will be avoided.

"""
import numpy as np 
#from collections import Counter
from scipy import stats

class FaissKNNImpl:
    
    def __init__(self,k,faiss):
        self.k = k # k nearest neighbor value
        self.faissIns = faiss # FAISS instance
        self.index = 0
        self.gpu_index_flat = 0 
        self.train_labels = []  
        self.test_label_faiss_output = [] 

        self.test_features_faiss_Index = 0 # hold the intermediate step from predict for predict_proba

    def sanitize_input(self, arr, function_name, convert_to_float=True):
        if type(arr) != "numpy.ndarray":
            arr = np.array(arr)

        if arr.flags["C_CONTIGUOUS"] is not True:
            print(f"Warning, a passed argument is not contiguous in function {function_name}. Using np.ascontiguousarray first. If you will be calling this often on large data, take care of this yourself.")
            # try:
            arr = np.ascontiguousarray(arr)
            # except Exception as e:
            #     raise Exception(e)
        if convert_to_float and not isinstance(arr, np.float32):
            arr = arr.astype('float32')

        return arr
        
    def fitModel(self,train_features,train_labels): 
        # Sanitize inputs first, or else faiss will throw difficult to understand errors.
        train_features = self.sanitize_input(train_features, "fitModel")
        train_labels = self.sanitize_input(train_labels, "fitModel", convert_to_float=False)
        
        if train_labels.flags["C_CONTIGUOUS"] is not True:
            train_labels = np.ascontiguousarray(train_labels)
            print("Warning, train_labels is not contiguous. Using np.ascontiguousarray first.")
        self.train_labels = train_labels
        self.index = self.faissIns.IndexFlatL2(train_features.shape[1])   # build the index 
        self.index.add(train_features)       # add vectors to the index
        
    def fitModel_GPU(self,train_features,train_labels):
        # Sanitize inputs first, or else faiss will throw difficult to understand errors.
        train_features = self.sanitize_input(train_features, "fitModel_GPU")
        train_labels = self.sanitize_input(train_labels, "fitModel_GPU", convert_to_float=False)

        no_of_gpus = self.faissIns.get_num_gpus()
        self.train_labels = train_labels
        self.gpu_index_flat = self.index = self.faissIns.IndexFlatL2(train_features.shape[1])   # build the index 
        if no_of_gpus > 0:
            self.gpu_index_flat = self.faissIns.index_cpu_to_all_gpus(self.index) 
            
        self.gpu_index_flat.add(train_features)       # add vectors to the index 
        return no_of_gpus
        
    def predict(self,test_features): 
        # Sanitize inputs first, or else faiss will throw difficult to understand errors.
        test_features = self.sanitize_input(test_features, "predict")

        distance, self.test_features_faiss_Index = self.index.search(test_features, self.k) 
        self.test_label_faiss_output = stats.mode(self.train_labels[self.test_features_faiss_Index],axis=1)[0]
        self.test_label_faiss_output = np.array(self.test_label_faiss_output.ravel())
        #for test_index in range(0,test_features.shape[0]):
        #    self.test_label_faiss_output[test_index] = stats.mode(self.train_labels[test_features_faiss_Index[test_index]])[0][0] #Counter(self.train_labels[test_features_faiss_Index[test_index]]).most_common(1)[0][0] 
        return self.test_label_faiss_output
    
    def predict_GPU(self,test_features):
        # Sanitize inputs first, or else faiss will throw difficult to understand errors.
        test_features = self.sanitize_input(test_features, "predict_GPU")

        distance, self.test_features_faiss_Index = self.gpu_index_flat.search(test_features, self.k) 
        self.test_label_faiss_output = stats.mode(self.train_labels[self.test_features_faiss_Index],axis=1)[0]
        self.test_label_faiss_output = np.array(self.test_label_faiss_output.ravel())
        return self.test_label_faiss_output

    # Handles both the multi-class and binary classification task very quickly. Could implement speed improvements for binary case, but would need to address label ordering in any case (which slows down many binary options).
    def predict_proba(self, test_features):
        # Sanitize inputs first, or else faiss will throw difficult to understand errors.
        test_features = self.sanitize_input(test_features, "predict_proba")

        votes = self.train_labels[self.test_features_faiss_Index]  # grab the votes (predicted label for each k) from CPU/GPU Predict
        unique_labels = np.unique(votes.ravel())
        # Create new (n_samples, unique.labels.length) array
        proba = np.zeros(shape=(votes.shape[0],unique_labels.shape[0]))

        idx = 0
        for rows in votes:
            unique_vals , unique_counts = np.unique(rows, return_counts=True)
            unique_probs = unique_counts / self.k
            proba_row = np.zeros(unique_labels.shape[0])

            for label in unique_vals:
                proba_row[np.argwhere(unique_labels == label)] = unique_probs[np.argwhere(unique_vals == label)]

            proba[idx] = proba_row
            idx += 1
        
        return proba, unique_labels  # returning unique_labels is important, as multiclass may be sorted oddly otherwise.
      
    def getAccuracy(self,test_labels):
        accuracy = (self.test_label_faiss_output == test_labels).mean() 
        return round(accuracy,2) 
