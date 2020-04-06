# Keith Brazill - G00387845
# Programming and Scripting Assignment April 2020
# Analysis of the Iris Flower Dataset using Python

#The following Libraries have been imported and used in the dataset. Please refer
#to readme for details on these libraries and why they are used.

import numpy as np #Used for arranging data and carrying out mathematical calculations
import pandas as pd #For reading and efficient management of the data
import seaborn as sns #Used for plotting in Seaborn module
import matplotlib.pyplot as plt #Used for plotting in Matplotlib.pyplot module
from pandas.plotting import parallel_coordinates #Used to specifically plot a parrallel co-ordinates plot

#Import the dataset. Using panda's we tell python to read the associated CSV text file.
iris = pd.read_csv('Iris_data.txt')
setosa=iris[iris['variety']=='Setosa']
versicolor=iris[iris['variety']=='Versicolor']
virginica=iris[iris['variety']=='Virginica']

#Following importing of the libraries the data set was imported by telling Pandas (pd) 
# to read the CSV file of the dataset. The different species and their locations in 
# the data set are also defined which are used further in the analysis.

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

#In the above code ".loc"[[]] allows us to only include the specific fields in the plot.
# The type of plot is an "area" (kind) and we define we the font size, figure size, 
# colormap and also that we want to append a table. Labels and titles are added and 
# the plot is saved as a figure. 

# 2 Boxplots
figboxplot, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(12,10)) 
sns.boxplot(x="variety", y="sepal.length", data=iris, ax=ax1)
sns.swarmplot(x="variety", y="sepal.length", data=iris, color="0.25", ax=ax1)
ax1.set_ylabel("Lenght/Width (cm)", color="g")
ax1.set_xlabel("Sepal Lenght (cm)", color='r')

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

# The box plot is combined by defining the axes to be included from the subplot and 
# the subplot size and number of rows and colums (1,4) 
# is defined. For each of the four variables a box plot and 
# swarmplot is appended using the sns library. Labels, axes, 
# data sets and colors are then defined. 

figboxplota=iris.boxplot(by="variety", figsize=(10, 8)) #alternative way to view boxplots
plt.savefig("3_IrisBoxPlotsB")
# Above code is an alternative method to show box plot, not as visually powerful as seaborn.

# 3 Jointplots
figjointplotsepal=sns.jointplot(x='sepal.length',y='sepal.width',data=iris, color="g")
plt.savefig("4_figjointplotsepal")
figjointplotpetal=sns.jointplot(x='petal.length',y='petal.width',data=iris, color="r")
plt.savefig("5_figjointplotpetal")
figjointplotsepalkde=sns.jointplot(x='sepal.length',y='sepal.width',data=iris, kind="kde", space=0, color='g')
plt.savefig("6_figjointplotsepalkde")
figjointplotpetalkde=sns.jointplot(x='sepal.length',y='sepal.width',data=iris, kind="kde", space=0, color='r')
plt.savefig("7_figjointplotpetalkde")

# Using the seaborn library these graphs are rather simple to form. we define the figure name 
# and tell sns to plot it as a joinplot with associated x and y labels, the data set to use 
# and the color map to use. For the kde format the kind "kde" is specified and the spacing 
# of the plot is defined. 

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

# The differences from the boxplot is that the winter color palette is used, 
# orientation is set to vertical, jitter is set to true (useful where data 
# overlaps it makes easier to see distribution) and edgecolor is set to 
# gray (color of the lines around each point).

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

# The individual scatterplots are same as above code , except the hue is 
# set to "variety" which gives a unique color for each species and adds a 
# legend for same. On these plots the x ticks were overlapping so the rotation 
# was modified to 90 degrees. On the y-axis there was only points every 0.5cm 
# which did not display enough data so the yticks were adjusted manually to 0.1cm intervals. 

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

# We tell python the figure size and the number of rows and columns and location 
# for each subplot (2,2,x). The x-axis is the species or "variety" and the y-axis 
# represents the attibute using the iris data set. Similar to all plots, this is 
# also saved to the repository. 

# 6 Pair Plot

figurepairplot = sns.pairplot(data=iris,kind='scatter', hue='variety')
plt.savefig("14_PairPlots")

# The seaborn pairplot is defined as figurepairplot, the iris data is used, 
# the kind of plot is a scatter and the plot is organised (hue) by the variety.

# 7 Heat Map
figirisheatmap, (ax13) = plt.subplots(figsize=(10,7)) 
ax13 =sns.heatmap(iris.corr(), annot=True, cmap='summer', vmin=-1, vmax=1, linewidths=1,linecolor='k',square=True)
ax13.set_ylim(0, 4) # explain the issue with the heatmap in matplotlib for credit
plt.savefig("15_IrisHeatMap")

# Heatmaps are a standard module in seaborn. The data used is the iris correlation data. 
# Setting the annot to "True" allows the values to be displayed within the boxes of the 
# heatmap. The color map used is summer, the range is set from -1 to +1, the linewidths 
# and line colors of the grid are defined and we want all data shown evenly by setting 
# square to "True". ***Note: there is a bug with using the seaborn heatmap in the 
# latest version of matplotlib, whereby the top and bottom of the plot is cropped. 
# To manually fix this issue the y limit of the axes was set (0,4) to display all the 
# data correctly.***

# 8 Distribution Plot

irisdist, axes = plt.subplots(2,2, figsize=(10,8), sharex=False)
sns.distplot(iris["sepal.length"], color='green', label="Sepal Length", ax=axes[0,0])
sns.distplot(iris["sepal.width"],color='red', label="Sepal Width", ax=axes[0,1])
sns.distplot(iris["petal.length"],color='blue', label="Petal Length", ax=axes[1,0])
sns.distplot(iris["petal.width"],color='gold', label="Petal Width", ax=axes[1,1])
plt.savefig("16_IrisDistPlot")

# The four sublplots are combined and each of the sub plots are given their own x axis. 
# We tell the plot to use the variable from the iris dataset for each of the subplots 
# and define the color, label and axes (location of the plot).

# 9 LM Plot

sns.lmplot(x="petal.length", y="petal.width", data=iris)
plt.savefig("17_IrisLMPlotPetal")
sns.lmplot(x="sepal.length", y="sepal.width", data=iris)
plt.savefig("18_IrisLMPlotSepal")

# Following early descriptions of code, the above is self explanotory. 
# The lmplot is another in build module to seaborn. 

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

# We define the plot axes, size and we want to share the x axis. 
# We previously defined the setosa, versicolor and virginica at the 
# start of the code and we use these functions to get the variable 
# for each species. The kind of plot is a cumulative histogram and 
# we use the bins, alpha (opacity) and density functions to format 
# the display of the plot. 

# 12 Parrallel Coordinates

irisparalledlcoord, axes = plt.subplots(1,1, figsize=(10,8))
parallel_coordinates(iris, "variety")
plt.savefig("21_IrisParrallelCoordinates")

# Parrallel coordinates is a module imported from Pandas library. 
# It is straightforward to plot using the data set and class. 
# It is a good graphical plot to show the variance for each of 
# the four variables with the biggest differences been the petal 
# length and width. 

# 13 Dashboard

dashboard, axes =plt.subplots(2,2, figsize=(15,15))
sns.set_style('darkgrid')
d1=sns.boxplot(x="variety", y="sepal.length", data=iris, ax=axes[0,0])
d2=sns.stripplot(x="variety", y="sepal.width", data=iris, palette="winter",size=5,jitter=True,edgecolor='gray',orient='v', ax=axes[0,1])
d3=sns.violinplot(x='variety',y='petal.length',data=iris, ax=axes[1,0])
d4=sns.distplot(iris["petal.width"],color='gold', label="Petal Width", ax=axes[1,1])
plt.savefig("22_IrisDashboard")

# The dashboard shows how we integrate different subplot types into one plot.

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

# The figure name is defined and the plot size. The data is taken from iris[variable] 
# and the kind of plot and color of the plot is defined. It is preferred to show the 
# grid in the histogram so this is set to "True" and finally image is automatically 
# saved to a unique name. 

# References
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














