#To run this file do
#1-From cmd type ipython
#2-Inside ipython CLI type: %matplotlib inline 
#3-Then type: %run ./ridgeRegression.py
#Import libraries for data and numeric handling
import pandas as pd
import numpy as np
#Import display libraries
from IPython.display import display
from IPython.html import widgets 
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual
import matplotlib.pyplot as plt
import seaborn as sns
#Import linear models
from sklearn.linear_model import LinearRegression
#Import model selection functions
from sklearn.model_selection import train_test_split
#Import polynomial features
from sklearn.preprocessing import PolynomialFeatures
#Import Ridge Regression
from sklearn.linear_model import Ridge
#Import Grid Search
from sklearn.model_selection import GridSearchCV

# Import clean data 
path = path='https://ibm.box.com/shared/static/q6iiqb1pd7wo8r3q28jvgsrprzezjqk3.csv'
df = pd.read_csv(path)

#GRID SEARCH
#===========
x_data=df.drop('price',axis=1)
y_data=df['price']
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)
#Define the parameters for the Ridge Model
parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000,10000,100000,100000]}]
#Crate the Ridge Model
RR=Ridge()
#Create the Grid Search and train it
Grid1 = GridSearchCV(RR, parameters1,cv=4)
Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']],y_data)
#Determine the best estimator
print(Grid1.best_estimator_)