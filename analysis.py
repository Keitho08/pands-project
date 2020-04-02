# Keith Brazill - G00387845
# Programming and Scripting Assignment April 2020
# Analysis of the Iris Flower Dataset using Python

#The following Libraries have been imported and used in the dataset. Please refer
#to readme for details on these libraries and why they are used.

import numpy as np
import pandas as pd
#import scipy
#import sklearn 
import seaborn as sns
import matplotlib.pyplot as plt

#from sklearn import metrics
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
#from matplotlib.backends.backend_pdf import PdfPages 
#pp = PdfPages('Irisplots.pdf')


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

iris.describe().loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].plot(kind = "area",fontsize=20, figsize = (20,8), table = True,colormap="rainbow")
plt.ylabel('Value', color="g", size=20)
plt.title("General Statistics of Iris Dataset", size=20)
plt.savefig("1_Iris_Data_Summary")

# 2 Boxplots
figboxplot, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(12,10)) 
sns.boxplot(x="variety", y="sepal.length", data=iris, ax=ax1)
sns.swarmplot(x="variety", y="sepal.length", data=iris, color="0.25", ax=ax1)
ax1.set_ylabel("Sepal Lenght (cm)", color="g")
ax1.set_xlabel("Length/Width (cm)", color='r')

sns.boxplot(x="variety", y="sepal.width", data=iris, ax=ax2)
sns.swarmplot(x="variety", y="sepal.width", data=iris, color="0.25", ax=ax2)
ax2.set_ylabel("")  
ax2.set_xlabel("Sepal Width", color='r')

sns.boxplot(x="variety", y="petal.length", data=iris, ax=ax3)
sns.swarmplot(x="variety", y="petal.length", data=iris, color="0.25", ax=ax3)
ax3.set_ylabel("")  
ax3.set_xlabel("Petal Length", color='r')

sns.boxplot(x="variety", y="petal.width", data=iris, ax=ax4)
sns.swarmplot(x="variety", y="petal.width", data=iris, color="0.25", ax=ax4)
ax4.set_ylabel("")  
ax4.set_xlabel("Petal Width", color='r')

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
ax13.set_ylim(0, 4)
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

# 13 Dashboard

dashboard, axes =plt.subplots(2,2, figsize=(15,15))
sns.set_style('darkgrid')
d1=sns.boxplot(x="variety", y="sepal.length", data=iris, ax=axes[0,0])
d2=sns.stripplot(x="variety", y="sepal.width", data=iris, palette="winter",size=5,jitter=True,edgecolor='gray',orient='v', ax=axes[0,1])
d3=sns.violinplot(x='variety',y='petal.length',data=iris, ax=axes[1,0])
d4=sns.distplot(iris["petal.width"],color='gold', label="Petal Width", ax=axes[1,1])
plt.savefig("22_IrisDashboard")

# 14 Individual Histograms

sepallengthhist, axes = plt.subplots(figsize=(10,8))
iris['sepal.length'].plot(kind='hist',color='blue')
plt.xlabel("Sepal Length")
plt.grid(True)
plt.savefig("23_sepallengthhist")

sepalwidthhist, axes = plt.subplots(figsize=(10,8))
iris['sepal.width'].plot(kind='hist',color='green')
plt.xlabel("Sepal Width")
plt.grid(True)
plt.savefig("24_sepalwidthhist")

petallengthhist, axes = plt.subplots(figsize=(10,8))
iris['petal.length'].plot(kind='hist',color='red')
plt.xlabel("Petal Length")
plt.grid(True)
plt.savefig("25_petallengthhist")

petalwidthhist, axes = plt.subplots(figsize=(10,8))
iris['petal.width'].plot(kind='hist',color='gold')
plt.xlabel("Petal Width")
plt.grid(True)
plt.savefig("26_petalwidthhist")

# References
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html
# https://stackoverflow.com/questions/52796015/using-corr-method-for-sklearn-bunch-object-iris
# https://seaborn.pydata.org/generated/seaborn.color_palette.html#seaborn.color_palette
# https://seaborn.pydata.org/generated/seaborn.stripplot.html
# https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Seaborn_Cheat_Sheet.pdf
# https://elitedatascience.com/python-seaborn-tutorial
# https://seaborn.pydata.org/tutorial/aesthetics.html
# https://statisticsbyjim.com/basics/probability-distributions/
# https://seaborn.pydata.org/tutorial/distributions.html
# https://medium.com/@neuralnets/data-visualization-with-python-and-seaborn-part-1-29c9478a8700
# http://www.lac.inpe.br/~rafael.santos/Docs/CAP394/WholeStory-Iris.html NBB
# https://www.kaggle.com/biphili/seaborn-matplotlib-plot-to-visualize-iris-data NBB
# https://www.datacamp.com/community/tutorials/seaborn-python-tutorial
# https://www.kaggle.com/jchen2186/machine-learning-with-iris-dataset
# https://www.ibm.com/cloud/blog/predictive-machine-learning-model-build-retrain
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# https://www.kaggle.com/sanniaf/basic-data-mining-methods-on-iris
# https://www.kaggle.com/sanniaf/basic-data-mining-methods-on-iris














