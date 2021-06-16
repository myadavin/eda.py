#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 17:50:49 2020

@author: myadav
"""

#===========================================================
#Objective : Explore any data and  Classification 
#===========================================================
#import required libraries
#===========================================================

import os 

import time

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier

#import reload

import warnings

warnings.filterwarnings("ignore")

#===========================================================
#Import Data
#===========================================================

file1 = input("Please enter the file name with extension :")

data_income = pd.read_csv(file1)

data = data_income.copy(deep=True)

"""
#===========================================================
# Data Exploration 
# 1. Getting to know data
# 2. Data pre processing / missing data
# 3. cross table and data validation and visualiazation
#===========================================================
"""
#===========================================================
# 1. Gettingto know data
#===========================================================
# for i in ( 1,2,3):
#     br = input('Exit :')
#     if br == 'exit':
#         break

# Check variable data type 
print(data.info())
input("Press Enter to continue...")

# Check for null data 
print('Data columns with null values\n', data.isnull().sum())
input("Press Enter to continue...")
 

# Summary of numerical variables

summary_num = data.describe()
print('\n Summary of numerical variables\n',summary_num)
input("Press Enter to continue...")

# Summary of Catogarical variables 
summary_cate = data.describe(include = 'O')
print('\n Summary of Catogarical var iables \n', summary_cate)
input("Press Enter to continue...")

# Frequency of each category and unique classes
for col in data.columns: 
    print(col) 
    print(' \n', col ,' \n', data[col].value_counts())
    print('\n ', col ,'  \n', np.unique(data[col]))
#    plt.hist(data[col])
#    plt.show()
#    time.sleep(2)
    sns.countplot(y=col, data=data)
#    time.sleep(2)
    plt.show()
    br = input("Press Enter to continue or 'exit' to exit... :")
    if br == 'exit':
        break


# See for any null value, and reimport as null
data = pd.read_csv(file1, na_values=[" ?"])


#===========================================================
# 2. Data pre processing / missing data
#===========================================================

print("Is there any null data? \n",data.isnull().sum())
input("Press Enter to continue...")


missing= data[data.isnull().any(axis=1)]

print(' Check Missing data in Variable Explorer')
input("Press Enter to continue...")

# Drop data if reuired 
data2 = data.dropna(axis=0)
correlation = data2.corr()
print(' Check Corelation data in Variable Explorer')
input("Press Enter to continue...")

#===========================================================
# 3. cross table and data validation
#===========================================================

# Bar diagram with counts for a HUE
print(data2.columns)
hue1 = input('Select HUE column :')

print('\n Column : ', hue1 ,' Unique values \n', np.unique(data2[hue1]))
hue1_one = input('Enter first value of Hue column : ')
hue1_two = input('Enter second value of Hue column : ')


for col in data2.columns: 
    print('Generating Bar graph for : ', col)
    sns.countplot(y=col, data=data, hue=hue1)
    plt.show()
    br = input("Press Enter to continue or 'exit' to exit... :")
    if br == 'exit':
        break


# Pair wise plot

for col in data2.columns: 
    print('Generating pair wise plot for : ', col)
    sns.pairplot(data2, kind='scatter', hue=col)
    plt.show()
    br = input("Press Enter to continue or 'exit' to exit... :")
    if br == 'exit':
        break
  
#===========================================================
# Logistic Regression
#===========================================================

# Reindexing salary status to 0 / 1 

data3=data2.copy(deep=True)
data3[hue1] = data3[hue1].map({hue1_one:0,hue1_two:1})
print(data3[hue1])

new_data = pd.get_dummies(data3, drop_first=True)

# Storing the colum name
column_list = list(new_data.columns)
print(column_list)

# Saperating input names from the list 
features = list(set(column_list)-set([hue1]))
print(features)


# Store the out put values in Y
y=new_data[hue1].values
print(y)

# Store input values in X
x=new_data[features].values
print(x)

# Split the data in train and test
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)

# Make an instance of model
LR=LogisticRegression()

# Fitting the values of x and y
LR.fit(train_x,train_y)

# Predictinhg  from Test data
prediction = LR.predict(test_x)

# Confusion matrix
confusion_matrix_result = confusion_matrix(test_y,prediction)
print(confusion_matrix_result)

# Calculate the accuracy 
accuracy_score = accuracy_score(test_y,prediction)
print('Accuracy Scire Logistic : ' , accuracy_score)

# Print misclassified values from prediction
print('Misclassified sample : %d ' % (test_y !=prediction).sum())

#===========================================================
# KNN
#===========================================================

#reload(np)

# Storing the K nearest neighbours classifier
KNN_classifier = KNeighborsClassifier()

# Fitting the values of x and y 
KNN_classifier.fit(train_x, train_y)

# Predict 
predictionKNN = KNN_classifier.predict(test_x)

# Performance metric check

confusion_matrixKNN = confusion_matrix(test_y,predictionKNN)
#print('\t','Predicted Values')
#print('Original Values','\n',confusion_matrixKNN)

# Calculating error value for 1 to knn_num

misclassified_sample = []
knn_num = input('Enter KNN neighbors : ')
for i in range(1,knn_num):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x,train_y)
    pred_i=knn.predict(test_x)
    misclassified_sample.append((test_y != pred_i).sum())  
    accuracy_scoreKNN = accuracy_score(test_y,predictionKNN)
    print('Accuracy Scire KNN : ' , accuracy_scoreKNN)
    
print(misclassified_sample)
