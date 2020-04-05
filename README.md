# Pands-Project 2020 
## Keith Brazill - G00387845
### Analysis of the Iris Flower Data Set using Python.

![22_IrisDashboard](22_IrisDashboard.png)
Dashboard of Dataset (source: Developed by Keith Brazill)

# Introduction

This README document outlines the learners (Keith Brazill) approach, methodology and analysis on the Iris Flower Data Set for the Programming and Scripting assignment 2020, as part of the requirements for partial fullfilment of a postgraduate diploma in Computer Science with Data Analytics. 

The README is structured as follows:
1. Background and Context: The history of the Iris Flower data set and its key attributes are described. This section also describes how this data set has been used in the field of computer science.
2. Analysis of the Iris Flower Data Set: The data set is analysed using Python. Summaries of the data are captued in text and visual data (plots) and more detailed analysis is then carried out on the data set and results are also outputted to text and visualisations. 
3. Machine Learning: Further analysis is carried out on the data set and machine learning algorithims are used to calculate the species type based on the inputted petal and sepal dimensions. 
4. Conculusion: Concluding remarks reflecting the learners findings and self reflection are included in the conclusion. 
5. References: Finally all references used in the analysis are included.


# 1. Background and Context

The Iris Flower Data Set is a multivariate type data set that was developed by Ronald Fisher in his paper the "The use of Multiple Measurements in Taxonomic Problems." The data set is well know in the field of pattern recognition literature and is very frequently referenced in modern day literature, particulary in a field of significant importance to the learner, which is machine learning. 
The data set contains three classes of fifty instances that each refer to a species type of the iris plant.

The key attribute information in the paper are as follows:

1. sepal length in cm
2. sepal width in cm
3. petal length in cm
4. petal width in cm
5. class:

     *Iris Setosa* | `Iris Versicolor` | **Iris Viriginica**
    --- | --- | ---
    ![Setosa](Iris_setosa.jpg) | ![Setosa](Iris_versicolor.jpg) | ![Setosa](Iris_virginica.jpg)
Iris Flower Species (source: https://en.wikipedia.org/wiki/Iris_flower_data_set) 
    
![sepalpetalimage](sepalpetalimage.png)

Iris Attributes (source: https://www.ritchieng.com/machine-learning-iris-dataset/)

The Iris Data Set is commonly referenced in computer science. The Iris flower data set is widely used as an input testing method for new types of sorting models and taxonomy algorithims to determine how various technologies can handle data sets. For example Wu et al. (2017) refered to the data set in their paper "Enhanced Classification Models for Iris Dataset," which focused on creating an induction algorithim for randomized decision trees. The dataset is widely used for beginners to understand machine learning, and it this is also an intention for this project to delve into the world of machine learning. A demonstration of the popularity of the dataset in machine learning is the fact that the widely used machine learning package Scikit-learn has the iris data set already built into it which can be accessed by the following python code:

```python
from sklearn.datasets import load_iris
```
The large American multinational technology company (IBM) which produces computer hardware, software and provides hosting and consulting services, have used the Iris flower data set to test many of their machine learning algorithims. 

# 2 Analysing the Iris Flower Data Set
## 2.1 Libraries

The first step in analysing the data is to import the relevant libraries required to visualise the data in Python. Several hours research was carried out to determine the appropriate libraries to use for this project.

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
```
All of the above libraries were pre-installed by Anaconda while installing Python. The following libaries were used:

* Numpy: Numpy is a library for python that supports arrays and hugh level mathematical functions. It is used in this project for mathematical calculations on the data and arranging the data numerically.
* Pandas: Pandas is a software library for Python for data manipulation and analysis. It is used in this project for reading files and managing data.
* Matplotlib.pyloy: Matplotlib is a library for plotting the python language and its extension Numpy. It is used for plotting the data in this project.
* Seaborn: Seaborn is a python data library to create visualisations based on matplotlib. Seaborn was specifically chosen for this project due to the attractive and informative graphical output.

## 2.2 The Data Set

Following importing of the libraries the data set was imported by telling Pandas (pd) to read the CSV file of the dataset. The different species and their locations in the data set are also defined which are used further in the analysis.

```python
iris = pd.read_csv('Iris_data.txt')
setosa=iris[iris['variety']=='Setosa']
versicolor=iris[iris['variety']=='Versicolor']
virginica=iris[iris['variety']=='Virginica']
```
The CSV file was sourced from the UCI Machine Learning Repository available at: http://archive.ics.uci.edu/ml/datasets/Iris. 

## 2.3 Summary of the Data

Following the loading of the CSV file we then want to get a high level understanding of the dataset. This is carried out by printing the head (the first 5 rows of the data), grouping the different species and count, the shape of the data (how many rows and columns in x,x format), a description of the data (count, mean, std, median, min, max and upper and lower quartiles) and the correlation between the different variables of the data set to a text file. 

```python
txt = open("00_Iris_Analysis_Output.txt", "w")
print(iris.head(5), file=txt)
print("The species of Iris are", iris.groupby('variety').size(), file=txt)
print(iris.shape, file=txt)
print(iris.describe(), file=txt)
print(iris.corr(), file=txt)
```
All of the output is automatically saved to the text file "00_Iris_Analysis_Output.txt" and is overwritten each time the program is ran. 

Head of the data shows each of the main attributes and format of the data:

```
 sepal.length  sepal.width  petal.length  petal.width variety
0           5.1          3.5           1.4          0.2  Setosa
1           4.9          3.0           1.4          0.2  Setosa
2           4.7          3.2           1.3          0.2  Setosa
3           4.6          3.1           1.5          0.2  Setosa
4           5.0          3.6           1.4          0.2  Setosa
```
The species area setosa, versicolor and virginica and there is 50 of each:

```
The species of Iris are variety
Setosa        50
Versicolor    50
Virginica     50
```
The shape of the data is 150 rows and 5 columns:
```
(150, 5)
```
The outputs of the description of the data are provided below:
```
sepal.length  sepal.width  petal.length  petal.width
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.057333      3.758000     1.199333
std        0.828066     0.435866      1.765298     0.762238
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000
```
And the correlation between the different variables is included below, it can be seen there is a very strong positive corelation between petal length and width, to be explained further in detailed analysis:
```
 sepal.length  sepal.width  petal.length  petal.width
sepal.length      1.000000    -0.117570      0.871754     0.817941
sepal.width      -0.117570     1.000000     -0.428440    -0.366126
petal.length      0.871754    -0.428440      1.000000     0.962865
petal.width       0.817941    -0.366126      0.962865     1.000000
```
Now that we have seen the summary of data in text format, it can also be reviewed graphically.

**Histograms of the Data**

We can display the description of the data using the (data).describe function:
![1_Iris_Data_Summary](1_Iris_Data_Summary.png)

This plot allows us to quickly visualise the main characteristics of the data. A table is appended to reference the values. By default .describe also includes the "count" of each attibute, however this made the other components extremely difficult to read, therefore the learner researched how to exclude this field to make the graph more understandable. The following code was used:
```python
iris.describe().loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].plot(kind = "area",fontsize=20, figsize = (20,8), table = True,colormap="rainbow")
plt.ylabel('Value', color="g", size=20)
plt.title("General Statistics of Iris Dataset", size=20)
plt.savefig("1_Iris_Data_Summary")
```
In the above code ".loc"[[]] allows us to only include the specific fields in the plot. The type of plot is an "area" (kind) and we define we the font size, figure size, colormap and also that we want to append a table. Labels and titles are added and the plot is saved as a figure. 

We can review the distribution of each of the key variables using Histogram plots.

![23_sepallengthhist](23_sepallengthhist.png)
![24_sepalwidthhist](24_sepalwidthhist.png)
![25_petallengthhist](25_petallengthhist.png)
![26_petalwidthhist](26_petalwidthhist.png)

The Histograms show us the range in centimetres(cm) for each of the variables and the frequency these occur in the dataset. The typical code used for the histograms is as follows:

```python
sepallengthhist, axes = plt.subplots(figsize=(10,8))
iris['sepal.length'].plot(kind='hist',color='blue')
plt.xlabel("Sepal Length")
plt.grid(True)
plt.savefig("23_sepallengthhist")
```
The figure name is defined and the plot size. The data is taken from iris[variable] and the kind of plot and color of the plot is defined. It is preferred to show the grid in the histogram so this is set to "True" and finally image is automatically saved to a unique name. 

Another useful way to get a graphic summary of the data is using Jointplots. These plots go a step further than the histograms as they allows us to compare the individual distribution and relationships between two variables:

![4_figjointplotsepal](4_figjointplotsepal.png)
![5_figjointplotpetal](5_figjointplotpetal.png)

On these plots the distributions are shown on the end of the top and side of the plot and a scatterplot is shown demonstrating the relationship between the sepal/petal lenghth and width. We can see the petal relationship is far more linear than the sepal.

An alternative method for showing the jointplots is in kde format. This is just an alternative way of showing the same data but it is graphically very powerful as the colormap of the plot reflects the data spread:

![6_figjointplotsepalkde](6_figjointplotsepalkde.png)
![7_figjointplotpetalkde](7_figjointplotpetalkde.png)

The code for the above plots is as follows:
```python
figjointplotpetal=sns.jointplot(x='petal.length',y='petal.width',data=iris, color="r")
plt.savefig("5_figjointplotpetal")
figjointplotsepalkde=sns.jointplot(x='sepal.length',y='sepal.width',data=iris, kind="kde", space=0, color='g')
plt.savefig("6_figjointplotsepalkde")
```
Using the seaborn library this graphs are rather simple to form. we define the figure name and tell sns to plot it as a joinplot with associated x and y labels, the data set to use and the color map to use. For the kde format the kind "kde" is specified and the spacing of the plot is defined. 

## 2.4 Detailed Analysis

The previous section provided a high level summary of the data, in the following section more detailed analysis is carried out on the data set to examine the relationship between the individual species in more detail, the relationship between different attributes and patterns in the data set. 

One of the most useful plots for looking at the breakdown of the characteritics between each of the species is the boxplot:

![2_IrisBoxPlotsA](2_IrisBoxPlotsA.png)

For the above particular boxplot a swarmplot was overlaid showing the distribution of the data (grey dots on the graph). The boxplot can be interpreted as follows:

![boxplotsexplained](boxplotsexplained.png)

Boxplot Interpretation (source:http://www.physics.csbsju.edu/stats/box2.html)

The grey dots that lie outside the the min or max. values are referred to as outliers, these are either above or below 1.5 times the inner quartile range (IQR). The boxplot is very useful to visually compare each of these box plots based on each species and variable and hence it was decided to compine these in one overall figure rather than four seperate outputs. For example, the boxplot shows us that the setosa has the smallest petal witdh, petal length and sepal length but it has the largest sepal width in comparison to the other species. 

The code use to show the boxplots is as follows:
```python
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
```
The box plot is combined by defining the axes to be included from the subplot and the subplot size and number of rows and colums (1,4) is defined. For each of the four variables a box plot and swarmplot is appended using the sns library. Labels, axes, data sets and colors are then defined. 

Similar to the boxplot, an overall scatterplot was created to show the overall distribution for each variable of each species:

![8_FigstripPlots](8_FigstripPlots.png)

The scatterplots are not as detailed as the box plots, however, for the average person these may be quickier and easier to interpret and may be more suited to other applications. The scatterplots are also useful to plot the two relationships against each other classified by species:

* Sepal Width vs Sepal Length
![9_FigstripplotSepalWvsL](9_FigstripplotSepalWvsL.png)

* Petal Width vs Petal Length
![10_FigstripplotPetalLvPW](10_FigstripplotPetalLvPW.png)

* Petal Length vs Sepal Length
![11_FigstripplotpetalLvsSL](11_FigstripplotpetalLvsSL.png)

* Sepal Width vs Petal Width
![12_FigstripplotpetalWvsSW](12_FigstripplotpetalWvsSW.png)

The code used for the scatterplots is similar to the boxplots, snips of the code is displayed below:
```python
figstripplot1, (ax5, ax6, ax7, ax8) = plt.subplots(1,4,figsize=(10,8)) 
sns.stripplot(x="variety",y="sepal.length",data=iris,palette="winter",ax=ax5,size=5,jitter=True,edgecolor='gray',orient='v')
ax5.set_xlabel("Sepal Lenght (cm)", color="g")
ax5.set_ylabel("(cm)", color="r")
plt.savefig("8_FigstripPlots")
```
The differences from the boxplot is that the winter color palette is used, orientation is set to vertical, jitter is set to true (useful where data overlaps it makes easier to see distribution) and edgecolor is set to gray (color of the lines around each point).


```python
figstripplotsepallwvssw, (ax9) = plt.subplots(1,1,figsize=(10,8)) 
sns.stripplot(x="sepal.length",y="sepal.width",data=iris,ax=ax9,size=5,jitter=True,edgecolor='gray',orient='v', hue="variety")
ax9.set_xticklabels(ax9.get_xticklabels(), rotation=90)
ax9.set_yticks(np.arange(1.5,5,0.1))
plt.savefig("9_FigstripplotSepalWvsL")
```
The individual scatterplots are same as above, except the hue is set to "variety" which gives a unique color for each species and adds a legend for same. On these plots the x ticks were overlapping so the rotation was modified to 90 degrees. On the y-axis there was only points every 0.5cm which did not display enough data so the yticks were adjusted manually to 0.1cm intervals. 

A combination of the scatterplot and boxplot is the Violin plot:

![13_violinplots](13_violinplots.png)

This plot allows us to visualise the distribution of the data and also the probability. The thick grey bar in the middle represents the IQR and the white dot represents the median. Violin plots do not offers as much functionaility as the boxplot or scatterplot as their are visually simplistic, but they do offer good graphical representation. The code used to generate the violin plot is as follows:

```python
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
```

We tell python the figure size and the number of rows and columns and location for each subplot (2,2,x). The x-axis is the species or "variety" and the y-axis represents the attibute using the iris data set. Similar to all plots, this is also saved to the repository. 

Pairplots are extremely useful for a quick review of all attributes in a data set against each other:

![14_PairPlots](14_PairPlots.png)

Pair plots are essentially scatterplots where one variable in the same data row is compared with other variables. In the above graph the y-axis is shared on each row and the x-axis is shared in each column. Each variable can be quickly compared for each species. For example we can see the versicolor consistently falls in the middle of the range, while the virginica has the largest attributes except for the sepal width. The code used for the pairplots is as below:

```python
figurepairplot = sns.pairplot(data=iris,kind='scatter', hue='variety')
plt.savefig("14_PairPlots")
```

The seaborn pairplot is defined as figurepairplot, the iris data is used, the kind of plot is a scatter and the plot is organised (hue) by the variety.

Another very useful plot is the HeatMap:

![15_IrisHeatMap](15_IrisHeatMap.png)




Machine Learning
Via the command line sklearn was installed using the following command: 
* pip install sklearn



# References

1. https://archive.ics.uci.edu/ml/datasets/Iris
2. Fisher,R.A. "The use of multiple measurements in taxonomic problems" Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to Mathematical Statistics" (John Wiley, NY, 1950).
3. https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet
4. https://www.techopedia.com/definition/32880/iris-flower-data-set
5. Yuanyuan Wu, Jing He, Yimu Ji, Guangli Huang, Haichang Yao, Peng Zhang, Wen Xu, Mengjiao Guo, Youtao Li,
Enhanced Classification Models for Iris Dataset,
Procedia Computer Science,
Volume 162,
2019,
Pages 946-954,
ISSN 1877-0509,
https://doi.org/10.1016/j.procs.2019.12.072.
(http://www.sciencedirect.com/science/article/pii/S1877050919320836)
6. https://scikit-learn.org/stable/user_guide.html
7. http://www.lac.inpe.br/~rafael.santos/Docs/CAP394/WholeStory-Iris.html
8. https://www.kaggle.com/sanniaf/basic-data-mining-methods-on-iris
9. https://www.ritchieng.com/machine-learning-iris-dataset/
10. https://www.ibm.com/cloud/blog/predictive-machine-learning-model-build-retrain
11. https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
12. https://python-graph-gallery.com/30-basic-boxplot-with-seaborn/
13. https://seaborn.pydata.org/tutorial/aesthetics.html
14. https://en.wikipedia.org/wiki/Pandas_(software)
15. https://en.wikipedia.org/wiki/NumPy
16. https://seaborn.pydata.org/
17. http://www.physics.csbsju.edu/stats/box2.html
