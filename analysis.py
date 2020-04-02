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
from pandas.plotting import parallel_coordinates
from matplotlib.backends.backend_pdf import PdfPages 
pp = PdfPages('Irisplots.pdf')


#Import the dataset. Using panda's we tell python to read the associated CSV text file.
iris = pd.read_csv('Iris_data.txt')
setosa=iris[iris['variety']=='Setosa']
versicolor=iris[iris['variety']=='Versicolor']
virginica=iris[iris['variety']=='Virginica']

#Summary of the data. 
#The first step in analysing the dataset is to get a brief summary
#of the data set.

txt = open("00_Iris_Analysis_Output.txt", "w")
print(iris.head(5), file=txt)
print("The species of Iris are", iris.groupby('variety').size(), file=txt)
print(iris.shape, file=txt)
print(iris.describe(), file=txt)
print(iris.corr(), file=txt)
txt.close()

#The data is re-written (w) to a text file included in the repository each time the program
#is run. The file is first opened (f=open) and then the head, variety, shape and description
#of the data is added to the text file. These outputs are explained further in the readme.

#Visualising the Data

# 1 Visual Description of the Data 

iris.describe().plot(kind = "area",fontsize=27, figsize = (20,8), table = True,colormap="rainbow")
plt.xlabel('Statistics', color="g")
plt.ylabel('Value', color="g")
plt.title("General Statistics of Iris Dataset")
plt.savefig("1_Iris_Data_Summary")

# 2 Boxplots
figboxplot, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,figsize=(8,6)) 
sns.boxplot(x="variety", y="sepal.length", data=iris, ax=ax1)
sns.swarmplot(x="variety", y="sepal.length", data=iris, color="0.25", ax=ax1)
ax1.set_ylabel("Sepal Lenght (cm)", color="g")

sns.boxplot(x="variety", y="sepal.width", data=iris, ax=ax2)
sns.swarmplot(x="variety", y="sepal.width", data=iris, color="0.25", ax=ax2)
ax2.set_ylabel("Sepal Width (cm)", color="g")  

sns.boxplot(x="variety", y="petal.length", data=iris, ax=ax3)
sns.swarmplot(x="variety", y="petal.length", data=iris, color="0.25", ax=ax3)
ax3.set_ylabel("Petal Lenght (cm)", color="g")  

sns.boxplot(x="variety", y="petal.width", data=iris, ax=ax4)
sns.swarmplot(x="variety", y="petal.width", data=iris, color="0.25", ax=ax4)
ax4.set_ylabel("Petal Width (cm)", color="g")  
plt.savefig("2_IrisBoxPlotsA")

figboxplota=iris.boxplot(by="variety", figsize=(10, 8)) #alternative way to view boxplots
plt.savefig("3_IrisBoxPlotsB")

# 3 Jointplots
figjointplotsepal=sns.jointplot(x='sepal.length',y='sepal.width',data=iris, color="g")
plt.savefig("4_figjointplotsepal")
figjointplotpetal=sns.jointplot(x='petal.length',y='petal.width',data=iris, color="r")
plt.savefig("5_figjointplotpetal")
figjointplotsepalkde=sns.jointplot(x='sepal.length',y='sepal.width',data=iris, kind="kde", space=0, color='g')
plt.savefig("6_figjointplotsepalkde")
figjointplotpetalkde=sns.jointplot(x='sepal.length',y='sepal.width',data=iris, kind="kde", space=0, color='r')
plt.savefig("7_figjointplotpetalkde")

# 4 Scatterplots
figstripplot1, (ax5, ax6, ax7, ax8) = plt.subplots(1,4,figsize=(10,8)) 
sns.stripplot(x="variety",y="sepal.length",data=iris,palette="winter",ax=ax5,size=5,jitter=True,edgecolor='gray',orient='v')
ax5.set_xlabel("Sepal Lenght (cm)", color="g")
ax5.set_ylabel("(cm)", color="r")

sns.stripplot(x="variety", y="sepal.width", data=iris, palette="winter", ax=ax6,size=5,jitter=True,edgecolor='gray',orient='v')
ax6.set_ylabel("Sepal Width (cm)", color="g")  
ax6.set_xlabel("Sepal Width (cm)", color="g")
ax6.set_ylabel("")

sns.stripplot(x="variety", y="petal.length", data=iris, palette="winter", ax=ax7,size=5,jitter=True,edgecolor='gray',orient='v')
ax7.set_ylabel("Petal Lenght (cm)", color="g")  
ax7.set_xlabel("Petal Lenght (cm)", color="g")
ax7.set_ylabel("")

sns.stripplot(x="variety", y="petal.width", data=iris, palette="winter", ax=ax8,size=5,jitter=True,edgecolor='gray',orient='v')
ax8.set_ylabel("Petal Width (cm)", color="g")  
ax8.set_xlabel("Petal Width (cm)", color="g")
ax8.set_ylabel("")
plt.savefig("8_FigstripPlots")

figstripplotsepallwvssw, (ax9) = plt.subplots(1,1,figsize=(10,8)) 
sns.stripplot(x="sepal.length",y="sepal.width",data=iris,ax=ax9,size=5,jitter=True,edgecolor='gray',orient='v', hue="variety")
ax9.set_xticklabels(ax9.get_xticklabels(), rotation=90)
ax9.set_yticks(np.arange(1.5,5,0.1))
plt.savefig("9_FigstripplotSepalWvsL")

figstripplotpetallwvpw, (ax10) = plt.subplots(1,1,figsize=(10,8)) 
sns.stripplot(x="petal.length",y="petal.width",data=iris,ax=ax10,size=5,jitter=True,edgecolor='gray',orient='v', hue="variety")
ax10.set_xticklabels(ax10.get_xticklabels(), rotation=90)
ax10.set_yticks(np.arange(0,3,0.1))
plt.savefig("10_FigstripplotPetalLvPW")

figstripplotpetallvsl, (ax11) = plt.subplots(1,1,figsize=(10,8)) 
sns.stripplot(x="petal.length",y="sepal.length",data=iris,ax=ax11,size=5,jitter=True,edgecolor='gray',orient='v', hue="variety")
ax11.set_xticklabels(ax11.get_xticklabels(), rotation=90)
ax11.set_yticks(np.arange(4,8.2,0.1))
plt.savefig("11_FigstripplotpetalLvsSL")

figstripplotpetallwvsw, (ax12) = plt.subplots(1,1,figsize=(10,8)) 
sns.stripplot(x="petal.width",y="sepal.width",data=iris,ax=ax12,size=5,jitter=True,edgecolor='gray',orient='v', hue="variety")
ax12.set_xticklabels(ax12.get_xticklabels(), rotation=90)
ax12.set_yticks(np.arange(1.5,4.6,0.1))
plt.savefig("12_FigstripplotpetalWvsSW")

# 5 Violin Plot

plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='variety',y='petal.length',data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='variety',y='petal.width',data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='variety',y='sepal.length',data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='variety',y='sepal.width',data=iris)
plt.savefig("13_violinplots")

# 6 Pair Plot

figurepairplot = sns.pairplot(data=iris,kind='scatter', hue='variety')
plt.savefig("14_PairPlots")

# 7 Heat Map
figirisheatmap, (ax13) = plt.subplots(figsize=(10,7)) 
ax13 =sns.heatmap(iris.corr(), annot=True, cmap='summer', vmin=-1, vmax=1, linewidths=1,linecolor='k',square=True)
plt.savefig("15_IrisHeatMap")

# 8 Distribution Plot

irisdist, axes = plt.subplots(2,2, figsize=(10,8), sharex=False)
sns.distplot(iris["sepal.length"], color='green', label="Sepal Length", ax=axes[0,0])
sns.distplot(iris["sepal.width"],color='red', label="Sepal Width", ax=axes[0,1])
sns.distplot(iris["petal.length"],color='blue', label="Petal Length", ax=axes[1,0])
sns.distplot(iris["petal.width"],color='gold', label="Petal Width", ax=axes[1,1])
plt.savefig("16_IrisDistPlot")

# 9 LM Plot

sns.lmplot(x="petal.length", y="petal.width", data=iris)
plt.savefig("17_IrisLMPlotPetal")
sns.lmplot(x="sepal.length", y="sepal.width", data=iris)
plt.savefig("18_IrisLMPlotSepal")

# 10 Cumulative Histogram

irishistsepallength, axes = plt.subplots(1,1, figsize=(10,8), sharex=True)
setosa['sepal.length'].plot(kind='hist',bins=200,alpha=0.3,color='blue',cumulative=True,density=True)
versicolor['sepal.length'].plot(kind='hist',bins=200,alpha=0.3,color='red',cumulative=True,density=True)
virginica['sepal.length'].plot(kind='hist',bins=200,alpha=0.3,color='green',cumulative=True,density=True)
plt.title('Sepal Length Distribution')
plt.legend(['Setosa','Versicolor','Virginica'])
plt.xlabel('Sepal Length in cm')
plt.axhline(0.75)
plt.axhline(0.5)
plt.axhline(0.25)
plt.savefig("19_CumulativeSepalLengthHistogram")

irishistpetallength, axes = plt.subplots(1,1, figsize=(10,8), sharex=True)
setosa['petal.length'].plot(kind='hist',bins=200,alpha=0.3,color='blue',cumulative=True,density=True)
versicolor['petal.length'].plot(kind='hist',bins=200,alpha=0.3,color='red',cumulative=True,density=True)
virginica['petal.length'].plot(kind='hist',bins=200,alpha=0.3,color='green',cumulative=True,density=True)
plt.title('Petal Length Distribution')
plt.legend(['Setosa','Versicolor','Virginica'])
plt.xlabel('Petal Length in cm')
plt.axhline(0.75)
plt.axhline(0.5)
plt.axhline(0.25)
plt.savefig("20_CumulativePetalLengthHistogram")

# 12 Parrallel Coordinates

irisparalledlcoord, axes = plt.subplots(1,1, figsize=(10,8))
parallel_coordinates(iris, "variety")
plt.ioff()
plt.savefig("21_IrisParrallelCoordinates")

# 12 Dashboard



# https://stackoverflow.com/questions/48204780/how-to-plot-multiple-figures-in-a-row-using-seaborn
