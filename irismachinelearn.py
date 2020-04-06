# Keith Brazill - G00387845
# Programming and Scripting Assignment April 2020
# Analysis of the Iris Flower Dataset using Python
# Additional Research - Machine Learning on Iris Data Set
# Using K Nearest Neighbors (KNN) algorithm

import pandas as pd 
import numpy as np #imports the load_iris function 
from sklearn.datasets import load_iris #imports the load_iris function 
from sklearn.model_selection import train_test_split #imports the function for creating the train/test split
from sklearn.neighbors import KNeighborsClassifier #Machine learning algorithim to be used
from sklearn import metrics #metrics module used to calculate accuracy of algorithim
import matplotlib.pyplot as plt 
iris = load_iris() # we used the inbuilt iris data in sklearn for the machine learning

# The sklearn was not previously installed so it was installed by running 
# "pip install sklearn" on the command line. The iris data set was loaded 
# from sklearn along with the required modules which will be explained 
# further in the README.

print(iris.feature_names) # Name of 4 features (column names)
print(iris.target) # Integers representing the species, 0=setosa, 1=versicolor, 2=virginica
print(iris.target_names) # Classes of target 
print(iris.data.shape) # 150 observations and 4 features
print(type(iris.data)) # class 'numpy.ndarray'
print(type(iris.target)) # class 'numpy.ndarray'

# splitting the data into training and test sets (80:20)
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.2, random_state=4)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print(X_train.shape) # prints shape of train data
print(X_test.shape) # prints shape of test data

# by using the sklearn module "train_test_split, using a test size as 20% of the data set. 
# We specify a random state number of 4 which tells the module for each run of the data the 
# split will always be the same and to use the same results, changing this value to 0 will mean 
# everytime the program is run it may yield a different result. It is not significant 
# what the a random_state number (4), the important thing is that everytime you use that 
# number (4) you will always get the same output.

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
plt.show()
print(knn.score(X_test, y_test))

# To verify that we have an 80/20 train/test split we can print the shape of each to the screen. 
# The n_neighbors and the reason why 5 is used is described further below, this is used to create 
# a good accuracy of the algorithim prediction.
# We use a range of 1 to 26 to test k, and using rhe metrics module of sklearn we can numerically 
# represent the accuracy for each k value. Using this code we can plot the relationship for the 
# values of K and the associated accuracy and use this to select the n_neighbors value for K in 
# the final algorithim.

classes = {0:'setosa', 1:'versicolor', 2:'virginica'}
sepallength = float(input("Sepal Length(cm).: ")) 
sepalwidth = float(input("Sepal Width(cm).: ")) 
petallength = float(input("Petal Length(cm).: ")) 
petalwidth = float(input("Petal Width(cm).: ")) 
X_new = np.array([[sepallength, sepalwidth, petallength, petalwidth]])
y_predict = knn.predict(X_new)
print(classes[y_predict[0]])

# The classes for each species, as described earlier, are defined. It is the intention of 
# this program to use it so that new measurements can be inputted for new research, hence 
# the user is prompted to input the values for each variable which is then included in a numpy 
# array. Using the knn predict function we can then predict the species type. The output is 
# a string printed to screen of the class type.

# References:
# 1. UCI 2020. Iris Flower Dataset. [online] Available at: <https://archive.ics.uci.edu/ml/datasets/Iris> [Accessed 6 April 2020].
# 2. Fisher,R.A. "The use of multiple measurements in taxonomic problems" Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to Mathematical Statistics" (John Wiley, NY, 1950).
# 3. GitHub. 2020. Adam-P/Markdown-Here. [online] Available at: <https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet> [Accessed 6 April 2020].
# 4. Techopedia.com. 2020. What Is The Iris Flower Data Set? - Definition From Techopedia. [online] Available at: <https://www.techopedia.com/definition/32880/iris-flower-data-set> [Accessed 6 April 2020].
# 5. Yuanyuan Wu, Jing He, Yimu Ji, Guangli Huang, Haichang Yao, Peng Zhang, Wen Xu, Mengjiao Guo, Youtao Li, Enhanced Classification Models for Iris Dataset,
# Procedia Computer Science, Volume 162, 2019, Pages 946-954, ISSN 1877-0509, (http://www.sciencedirect.com/science/article/pii/S1877050919320836).
# 6. Scikit-learn.org. 2020. User Guide: Contents — Scikit-Learn 0.22.2 Documentation. [online] Available at: <https://scikit-learn.org/stable/user_guide.html> [Accessed 6 April 2020].
# 7. Lac.inpe.br. 2020. Data Science Example - Iris Dataset. [online] Available at: <http://www.lac.inpe.br/~rafael.santos/Docs/CAP394/WholeStory-Iris.html> [Accessed 6 April 2020].
# 8. Kaggle.com. 2020. Basic Data Mining Methods On IRIS. [online] Available at: <https://www.kaggle.com/sanniaf/basic-data-mining-methods-on-iris> [Accessed 6 April 2020].
# 9. ritchieng.github.io. 2020. Iris Dataset. [online] Available at: <https://www.ritchieng.com/machine-learning-iris-dataset/> [Accessed 6 April 2020].
# 10. Ibm.com. 2020.Us-En_Cloud_Blog_Predictive-Machine-Learning-Model-Build-Retrain. [online] Available at: <https://www.ibm.com/cloud/blog/predictive-machine-learning-model-build-retrain> [Accessed 6 April 2020].
# 11. Brownlee, J., 2020. Your First Machine Learning Project In Python Step-By-Step. [online] Machine Learning Mastery. Available at: <https://machinelearningmastery.com/machine-learning-in-python-step-by-step/> [Accessed 6 April 2020].
# 12. Holtz, Y., 2020. #30 Basic Boxplot With Seaborn. [online] The Python Graph Gallery. Available at: <https://python-graph-gallery.com/30-basic-boxplot-with-seaborn/> [Accessed 6 April 2020].
# 13. Seaborn.pydata.org. 2020. Controlling Figure Aesthetics — Seaborn 0.10.0 Documentation. [online] Available at: <https://seaborn.pydata.org/tutorial/aesthetics.html> [Accessed 6 April 2020].
# 14. En.wikipedia.org. 2020. Pandas (Software). [online] Available at: <https://en.wikipedia.org/wiki/Pandas_(software)> [Accessed 6 April 2020].
# 15. En.wikipedia.org. 2020. Numpy. [online] Available at: <https://en.wikipedia.org/wiki/NumPy> [Accessed 6 April 2020].
# 16. Seaborn.pydata.org. 2020. Seaborn: Statistical Data Visualization — Seaborn 0.10.0 Documentation. [online] Available at: <https://seaborn.pydata.org/> [Accessed 6 April 2020].
# 17. Physics.csbsju.edu. 2020. Box Plot: Display Of Distribution. [online] Available at: <http://www.physics.csbsju.edu/stats/box2.html> [Accessed 6 April 2020].
# 18. Medium. 2020. K Nearest Neighbors And Implementation On Iris Data Set. [online] Available at: <https://medium.com/@avulurivenkatasaireddy/k-nearest-neighbors-and-implementation-on-iris-data-set-f5817dd33711> [Accessed 6 April 2020].
# 19. Medium. 2020. KNN Using Scikit-Learn. [online] Available at: <https://towardsdatascience.com/knn-using-scikit-learn-c6bed765be75> [Accessed 6 April 2020].