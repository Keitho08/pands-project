# Pands-Project 2020 
## Keith Brazill - G00387845
### Analysis of the Iris Flower Data Set using Python.

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
    

The Iris Data Set is commonly referenced in computer science. The Iris flower data set is widely used as an input testing method for new types of sorting models and taxonomy algorithims to determine how various technologies can handle data sets. For example Wu et al. (2017) refered to the data set in their paper "Enhanced Classification Models for Iris Dataset," which focused on creating an induction algorithim for randomized decision trees. The dataset is widely used for beginners to understand machine learning, and it this is also an intention for this project to delve into the world of machine learning. A demonstration of the popularity of the dataset in machine learning is the fact that the widely used machine learning package Scikit-learn has the iris data set already built into it which can be accessed by the following python code:

```python
from sklearn.datasets import load_iris
```
The large American multinational technology company (IBM) which produces computer hardware, software and provides hosting and consulting services, have used the Iris flower data set to test many of their machine learning algorithims. 

# 2 Analysing the Iris Flower Data Set
## 2.1 Summary of the Data

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
* Seaborn: 


The first step in analysing the data is to import the relevant libraries required to visualise the data in Python. Several hours research was carried out to determine the appropriate libraries to use for this project.


![sepalpetalimage](sepalpetalimage.png)

Source: https://www.ritchieng.com/machine-learning-iris-dataset/


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
