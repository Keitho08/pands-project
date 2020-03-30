# Keith Brazill - G00387845
# Programming and Scripting Assignment April 2020
# Analysis of the Iris Flower Dataset using Python

#The following Libraries have been imported and used in the dataset. Please refer
#to readme for details on these libraries and why they are used.

import numpy as np
import pandas as pd
import scipy
import sklearn 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_pdf import PdfPages 
pp = PdfPages('Irisplots.pdf')


#Import the dataset. Using panda's we tell python to read the associated CSV text file.
iris = pd.read_csv('Iris_data.txt')

#Summary of the data. 
#The first step in analysing the dataset is to get a brief summary
#of the data set.

txt = open("Iris_Analysis_Output.txt", "w")
print(iris.head(5), file=txt)
print("The species of Iris are", iris.groupby('variety').size(), file=txt)
print(iris.shape, file=txt)
print(iris.describe(), file=txt)
txt.close()

#The data is re-written (w) to a text file included in the repository each time the program
#is run. The file is first opened (f=open) and then the head, variety, shape and description
#of the data is added to the text file. These outputs are explained further in the readme.

#Visualising the Data

# Boxplots
# iris.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()

# figa, ax = plt.subplots(figsize=(10,8))
# ax = sns.boxplot(x="variety", y="sepal.length", data=iris)
# ax = sns.swarmplot(x="variety", y="sepal.length", data=iris, color="0.25")

# figb, ax = plt.subplots(figsize=(10,8))
# ax = sns.boxplot(x="variety", y="sepal.width", data=iris)
# ax = sns.swarmplot(x="variety", y="sepal.width", data=iris, color="0.25")

# figc, ax = plt.subplots(figsize=(10,8))
# ax = sns.boxplot(x="variety", y="petal.length", data=iris)
# ax = sns.swarmplot(x="variety", y="petal.length", data=iris, color="0.25")

# figd, ax = plt.subplots(figsize=(10,8))
# ax = sns.boxplot(x="variety", y="petal.width", data=iris)
# ax = sns.swarmplot(x="variety", y="petal.width", data=iris, color="0.25")

# pp.savefig(figa)
# pp.savefig(figb)
# pp.savefig(figc)
# pp.savefig(figd)
# pp.close()

figboxplot, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,figsize=(8,6))
#title= "Comparison of Iris Flower Species Sepal and Petal Sizes"
#plt.title(title, fontsize=15, color="b", loc="left")

sns.boxplot(x="variety", y="sepal.length", color="r", data=iris, ax=ax1)
sns.swarmplot(x="variety", y="sepal.length", data=iris, color="0.25", ax=ax1)
ax1.set_ylabel("Sepal Lenght (cm)", color="g")

sns.boxplot(x="variety", y="sepal.width", data=iris, ax=ax2)
sns.swarmplot(x="variety", y="sepal.width", data=iris, color="0.25", ax=ax2)
ax2.set_ylabel("Sepal Width (cm)", color="g")  # remove y label, but keep ticks

sns.boxplot(x="variety", y="petal.length", data=iris, ax=ax3)
sns.swarmplot(x="variety", y="petal.length", data=iris, color="0.25", ax=ax3)
ax3.set_ylabel("Petal Lenght (cm)", color="g")  # remove y label, but keep ticks

sns.boxplot(x="variety", y="petal.width", data=iris, ax=ax4)
sns.swarmplot(x="variety", y="petal.width", data=iris, color="0.25", ax=ax4)
ax4.set_ylabel("Petal Width (cm)", color="g")  # remove y label, but keep ticks

plt.show()
pp.savefig(figboxplot)
pp.close()


# 1 Sepal Lenght

# tmp = iris.drop("sepal.width", "sepal.length", "petal.width", "petal.length", axis=1)
# g = sns.boxplot(tmp, hue='Variety', markers='o')
# plt.show()


# sns.set(style="whitegrid", palette="Set3")
# title="Comparison of Iris Flower Species Sepal Length"
# sns.boxplot(x="variety", y="sepal.length", data=iris)
# ax = sns.swarmplot(x="variety", y="sepal.length", data=iris, color=".25")
# increasing font size
# plt.title(title, fontsize=15, color="b")
# Show the plot
# plt.savefig("Boxplot_Sepal_Lenght")
# plt.show()
# imagea.close()

# # 2 Sepal Width
# sns.set(style="whitegrid", palette="Set3")
# title="Comparison of Iris Flower Species Sepal Width"
# sns.boxplot(x="variety", y="sepal.width", data=iris)
# ax = sns.swarmplot(x="variety", y="sepal.width", data=iris, color=".25")
# # increasing font size
# plt.title(title, fontsize=15, color="b")
# # Show the plot
# #plt.savefig("Boxplot_Sepal_Width")
# #imageb.close()
# plt.show()

# https://stackoverflow.com/questions/48204780/how-to-plot-multiple-figures-in-a-row-using-seaborn
