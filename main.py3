# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

from sklearn import datasets
iris = datasets.load_iris()


iris_data = iris.data
iris_data = pd.DataFrame(iris_data, columns = iris.feature_names)
iris_data['class'] = iris.target
print(iris_data.head())

print(iris.target_names)


print(iris_data.shape)


#Now we will try different Machine Learning algorithms on the data to check for the accuracy scores of the different models

#Train Test Split

from sklearn.model_selection import train_test_split
X = iris_data.values[:,0:4]
Y = iris_data.values[:,4]
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.3,random_state = 42)


# At first we will try the K-Neighbors Classifier
model = KNeighborsClassifier()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print('Accuracy using K-Neighbors Classifier = ',accuracy_score(y_test,predictions))

# Now we will try the SVC algorithm

model = SVC()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print('Accuracy using SVC = ',accuracy_score(y_test,predictions))

# Now we will try the Random Forest Classifier

model = RandomForestClassifier(n_estimators = 5)
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print('Accuracy using Random Forest Classifier = ',accuracy_score(y_test,predictions))


# Now we will use the Logistic Regression model

model = LogisticRegression()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print('Accuracy using Logistic Regression = ',accuracy_score(y_test,predictions))
