# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("Machine_Breadown.csv")
df.head()

"""
  Column 'Has broken down in the past year' has 'Yes' and 'yes'
  as 2 separate values - may cause problem when creating
  dummies

  Fixing that in this block
"""

column = "Has_broken_down_in_the_past_year"
# print(df[column].unique())
df.loc[df[column] == "yes", column] = "Yes"

# Getting dummy variables for categorical data

df = pd.get_dummies(
    df, 
    columns= ["Age_of_Machine", "Severity", "Cause"], 
    drop_first= True, prefix_sep= ":"
)
df.head()

# Converting 'Cause' to a numeric value

encoder = LabelEncoder()
df['Cause'] = encoder.fit_transform(df['Cause'])
df.head()

X = df.drop(columns= ['Long_Term_Breakdown'])
y = df['Long_Term_Breakdown']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 0)

# Standardize features

obj = StandardScaler()

X_train = obj.fit_transform(X_train)
X_test  = obj.transform(X_test)

y_train = y_train.to_numpy()
y_test  = y_test.to_numpy()

# Add bias column

X_train = np.append(X_train, np.ones((X_train.shape[0], 1)), axis= 1)
X_test  = np.append(X_test,  np.ones((X_test.shape[0],  1)), axis= 1)

# Gradient Function
def gradient_descent(X, y, iterations= 3000, learningRate= 0.1):
    def take_downward_step():
        nonlocal m
        N = X.shape[0]

        m_t = m.copy().reshape((X.shape[1], 1))
        A = np.matmul(X, m_t)
        y_pred = 1/(1 + np.exp(-A))
        y_pred = y_pred.reshape(y.shape)

        err = (-learningRate/N) * (y - y_pred)

        for i in range(X.shape[0]):
            m -= (err[i] * X[i])

    m = np.zeros(X.shape[1])
    for i in range(1, iterations + 1):
        take_downward_step()

        if i % 100 == 0:
          print("Iterations done: {}".format(i))

    return m

m = gradient_descent(X_train, y_train)

def predict(m, X):
    N = X.shape[0]
   
    y_pred = []
    for i in range(N):
        y = (m * X[i]).sum()
        h = 1/(1 + math.exp(-y))
        y_pred.append(h)

    y_pred = np.array(y_pred)
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0

    return y_pred

y_pred = predict(m, X_test)

def accuracy(y_pred, y_test):
    t = (y_pred == y_test)
    return (t == True).sum()/len(t)

print("Accuracy obtained: {}".format(accuracy(y_pred, y_test)))

# Logistic Regression using sklearn library (to compare results)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train, y_train)
print("Accuracy obtained: {}".format(clf.score(X_test, y_test)))

