# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 22:32:28 2023

@author: AutoLab
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from tqdm import tqdm 
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.calibration import CalibratedClassifierCV
import time
#%%
def read_colorHist(path):
    with open(path) as f:
        label_list = []
        color_hist = np.empty((0,768))
        temp_hist = np.empty((0,768))
        for line in tqdm(f.readlines()):
            s = line.split(' ')
            print(s)
            img = cv2.imread(s[0])
            img = cv2.resize(img, (128, 128))
            # Calculate histogram without mask
            hist1 = cv2.calcHist([img],[0],None,[256],[0,256])
            hist2 = cv2.calcHist([img],[1],None,[256],[0,256])
            hist3 = cv2.calcHist([img],[2],None,[256],[0,256])
            hist = np.concatenate((hist1.T,hist2.T,hist3.T),axis=1)

            temp_hist = np.concatenate((temp_hist,hist),axis=0)

            if len(temp_hist) > 1000:
                color_hist = np.concatenate((color_hist,temp_hist),axis=0)
                temp_hist = np.empty((0,768))
                
            label_list.append(s[1][:-1])
        color_hist = np.concatenate((color_hist,temp_hist),axis=0)
        label_list = np.array(label_list)
        label_list = np.reshape(label_list, (len(label_list),1))
    return color_hist, label_list

def top1_5_score(true_y,prd_y,model):
    label = model.classes_
    top1_score = 0
    top5_score = 0
    for i in range(len(true_y)):
        top5_ans = np.argpartition(prd_y[i], -5)[-5:]
        if str(int(true_y[i])) in label[top5_ans]:
            top5_score = top5_score + 1
        if str(int(true_y[i])) == label[np.argmax(prd_y[i])]:
            top1_score = top1_score + 1
    return top1_score/len(true_y) , top5_score/len(true_y)

def plot_accuracy_curve(train_sizes,train_scores, valid_scores):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)
    plt.figure(figsize=(12,5))
    plt.title('')
    plt.plot(train_sizes,train_mean,'o-',color='r',label ='Training Score')
    plt.plot(train_sizes,valid_mean,'o-',color='g',label ='Validation Score')
    plt.fill_between(train_sizes, train_mean-train_std,train_mean+train_std,alpha=0.15,color='r')
    plt.fill_between(train_sizes, valid_mean-valid_std,valid_mean+valid_std,alpha=0.15,color='g')
    plt.legend()
    plt.xlabel("Training Data Sizes")
    plt.ylabel("Accuracy")
    plt.ylim([-0.05,1.05])
    plt.show()
#%%
train_x, train_y = read_colorHist("train.txt")
val_x, val_y = read_colorHist("val.txt")
test_x, test_y = read_colorHist("test.txt")
#%%
from sklearn.model_selection import learning_curve
train_sizes_x, train_scores_x, valid_scores_x = learning_curve(XGBClassifier(), train_x, train_y, train_sizes=[10000,20000, 30000, 40000], cv=3)
train_sizes_s, train_scores_s, valid_scores_s = learning_curve(svm.SVC(kernel='rbf',C=1,gamma='auto'), train_x, train_y, train_sizes=[10000,20000, 30000, 40000], cv=3)
train_sizes_p, train_scores_p, valid_scores_p = learning_curve(Perceptron(tol=1e-3, random_state=0), train_x, train_y, train_sizes=[10000,20000, 30000, 40000], cv=3)
#%%
plot_accuracy_curve(train_sizes_x, train_scores_x, valid_scores_x)
plot_accuracy_curve(train_sizes_s, train_scores_s, valid_scores_s)
plot_accuracy_curve(train_sizes_p, train_scores_p, valid_scores_p)
#%%
xgb = XGBClassifier()
start_time = time.time()
xgb = xgb.fit(train_x, train_y)
print("--- %s seconds --- XGB" % (time.time() - start_time))
y_pre_val_x = xgb.predict_proba(val_x)
#%%
svm_Model=svm.SVC(kernel='rbf',C=1,gamma='auto')
supvm = CalibratedClassifierCV(svm_Model, cv=3, method='isotonic')
start_time = time.time()
supvm.fit(train_x, train_y)
print("--- %s seconds --- SVM" % (time.time() - start_time))
y_pre_val_s = supvm .predict_proba(val_x)
top1_5_score(val_y, y_pre_val_s,supvm)
#%%
per = Perceptron(tol=1e-3, random_state=0)
pct = CalibratedClassifierCV(per, cv=3, method='isotonic')
start_time = time.time()
pct.fit(train_x, train_y)
print("--- %s seconds --- Perceptron" % (time.time() - start_time))
y_pre_val_p = pct.predict_proba(val_x)
y_pre_valp= pct.predict(val_x)
top1_5_score(val_y, y_pre_val_p,pct)
#%%
print("------------------------------Validation Results-------------------------------")
y_pre_val_x = xgb.predict_proba(val_x)
y_pre_val_s = supvm.predict_proba(val_x)
y_pre_val_p = pct.predict_proba(val_x)
xgb_results = top1_5_score(val_y, y_pre_val_x,xgb)
svm_results = top1_5_score(val_y, y_pre_val_s,supvm)
pct_results = top1_5_score(val_y, y_pre_val_p,pct)
print("XGB TOP-1 Accuracy :" + str(xgb_results[0]) + ", TOP-5 Accuracy :" + str(xgb_results[1]))
print("SVM TOP-1 Accuracy :" + str(svm_results[0]) + ", TOP-5 Accuracy :" + str(svm_results[1]))
print("Perception TOP-1 Accuracy :" + str(pct_results[0]) + ", TOP-5 Accuracy :" + str(pct_results[1]))
print("------------------------------Testing Results-------------------------------")
y_pre_test_x = xgb.predict_proba(test_x)
y_pre_test_s = supvm.predict_proba(test_x)
y_pre_test_p = pct.predict_proba(test_x)
xgb_results = top1_5_score(test_y, y_pre_test_x,xgb)
svm_results = top1_5_score(test_y, y_pre_test_s,supvm)
pct_results = top1_5_score(test_y, y_pre_test_p,pct)
print("XGB TOP-1 Accuracy :" + str(xgb_results[0]) + ", TOP-5 Accuracy :" + str(xgb_results[1]))
print("SVM TOP-1 Accuracy :" + str(svm_results[0]) + ", TOP-5 Accuracy :" + str(svm_results[1]))
print("Perception TOP-1 Accuracy :" + str(pct_results[0]) + ", TOP-5 Accuracy :" + str(pct_results[1]))

