# Keith Brazill - G00387845
# Programming and Scripting Assignment April 2020
# Analysis of the Iris Flower Dataset using Python
# Additional Research - Machine Learning on Iris Data Set
# Using K Nearest Neighbors (KNN) algorithm

import pandas as pd 
import numpy as np #imports the load_iris function 
from sklearn.datasets import load_iris #imports the load_iris function 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics
import matplotlib.pyplot as plt
iris = load_iris()

print(iris.feature_names) # Name of 4 features (column names)
# print(iris.target) # Integers representing the species, 0=setosa, 1=versicolor, 2=virginica
# print(iris.target_names) # Classes of target 
# print(iris.data.shape) # 150 observations and 4 features
# print(type(iris.data)) # class 'numpy.ndarray'
# print(type(iris.target)) # class 'numpy.ndarray'

# splitting the data into training and test sets (80:20)
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.2, random_state=4)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
# print(X_train.shape)
# print(X_test.shape)

k_range = range(1, 26)
scores = {}
scores_list = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred=knn.predict(X_test)
    scores[k] = metrics.accuracy_score(y_test, y_pred)
    scores_list.append(metrics.accuracy_score(y_test, y_pred))
plt.plot(k_range, scores_list)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
#plt.show()
#print(knn.score(X_test, y_test))

classes = {0:'setosa', 1:'versicolor', 2:'virginica'}
sepallength = float(input("Sepal Length(cm).: ")) 
sepalwidth = float(input("Sepal Width(cm).: ")) 
petallength = float(input("Petal Length(cm).: ")) 
petalwidth = float(input("Petal Width(cm).: ")) 
X_new = np.array([[sepallength, sepalwidth, petallength, petalwidth]])
y_predict = knn.predict(X_new)
print(classes[y_predict[0]])

# References:
# https://medium.com/@avulurivenkatasaireddy/k-nearest-neighbors-and-implementation-on-iris-data-set-f5817dd33711
# https://medium.com/@avulurivenkatasaireddy/k-nearest-neighbors-and-implementation-on-iris-data-set-f5817dd33711
# https://www.ibm.com/cloud/blog/predictive-machine-learning-model-build-retrain