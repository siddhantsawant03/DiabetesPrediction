# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:10:50 2024

@author: Sisa
"""

# IMPORTING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import accuracy_score

##################       1
# LOADING THE DF
diabetes_df = pd.read_csv('diabetes.csv')
print(diabetes_df.head())

diabetes_df.shape
diabetes_df.describe()
diabetes_df['Outcome'].value_counts()

##################       2

# MAIN GIST OF THE PREDICTION
diabetes_df.groupby('Outcome').mean()

X = diabetes_df.drop(columns='Outcome', axis=1)
Y = diabetes_df['Outcome']

print(X)
print(Y)

# Store column names before scaling
feature_names = X.columns.tolist()

# PLOTTING
p = diabetes_df.hist(figsize=(20, 20))

##################       3

# STANDARDIZATION OF DATA
scaler = StandardScaler()
scaler.fit(X)
std_data = scaler.transform(X)

print(std_data)

X = std_data

##################       4

# SPLITTING DATA INTO TESTING AND TRAINING DATA
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

##################       5

# TRAINING THE MODEL
classifier = svm.SVC(kernel='linear')

classifier.fit(X_train, Y_train)

# MODEL EVALUATION
X_train_prediction = classifier.predict(X_train)
training_data_acc = accuracy_score(X_train_prediction, Y_train)
print("Training Accuracy:", training_data_acc)

X_test_prediction = classifier.predict(X_test)
testing_data_acc = accuracy_score(X_test_prediction, Y_test)
print("Testing Accuracy:", testing_data_acc)

##################       6

# Making A PREDICTIVE SYSTEM

# input_data = (4,110,92,0,0,37.6,0.191,30)
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

input_data_as_np_array = np.asarray(input_data)

# reshaping the input data
input_data_reshaped = input_data_as_np_array.reshape(1, -1)

# standardizing the input data
std_input_data = scaler.transform(input_data_reshaped)

prediction = classifier.predict(std_input_data)
print("Prediction:", prediction)

if prediction[0] == 1:
    print("THE PERSON IS DIABETIC")
else:
    print("THE PERSON IS NOT DIABETIC")

##################       7

# Plotting Heatmap
plt.figure(figsize=(12, 10))
p = sns.heatmap(diabetes_df.corr(), annot=True, cmap='RdYlGn')
plt.title('Correlation Heatmap')
plt.show()

##################       8

# Plotting feature importances
rfc_model = RandomForestClassifier(n_estimators=100, random_state=2)
rfc_model.fit(X_train, Y_train)

plt.figure(figsize=(10, 8))
(pd.Series(rfc_model.feature_importances_, index=feature_names)
   .nlargest(10)  # Selecting top 10 features
   .plot(kind='barh'))
plt.title('Top 10 Important Features')
plt.xlabel('Feature Importance')
plt.show()
