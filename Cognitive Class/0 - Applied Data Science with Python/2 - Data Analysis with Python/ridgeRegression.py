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

# Import clean data 
path = path='https://ibm.box.com/shared/static/q6iiqb1pd7wo8r3q28jvgsrprzezjqk3.csv'

df = pd.read_csv(path)

#First lets only use numeric data
df=df._get_numeric_data()

#Functions for plotting
#Distribution plot
def DistributionPlot(RedFunction,BlueFunction,RedName,BlueName,Title,FileName):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)
    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')
    plt.show()
    plt.savefig(FileName)
#Polynomial Regression Plot
#xtrain,y_train: training data 
#xtesty_test:testing data 
#lr:  linear regression object 
#poly_transform:  polynomial transformation object 
def PollyPlot(xtrain,xtest,y_train,y_test,lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    xmax=max([xtrain.values.max(),xtest.values.max()])
    xmin=min([xtrain.values.min(),xtest.values.min()])
    x=np.arange(xmin,xmax,0.1)
    plt.plot(xtrain,y_train,'ro',label='Training Data')
    plt.plot(xtest,y_test,'go',label='Test Data')
    plt.plot(x,lr.predict(poly_transform.fit_transform(x.reshape(-1,1))),label='Predicted Function')
    plt.ylim([-10000,60000])
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('polyplot.png')

#RIDGE REGRESSION
#================
x_data=df.drop('price',axis=1)
y_data=df['price']
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)
pr=PolynomialFeatures(degree=2)
a=x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']]
b=x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']]
x_train_pr=pr.fit_transform(a)
x_test_pr=pr.fit_transform(b)
#Create and train the Ridge Model
RigeModel=Ridge(alpha=0.1)
RigeModel.fit(x_train_pr,y_train)
#We select the value of alpha that will minimize the error of the model
Rsqu_test=[]
Rsqu_train=[]
dummy1=[]
ALFA=5000*np.array(range(0,10000))
for alfa in ALFA:
    RigeModel=Ridge(alpha=alfa) 
    RigeModel.fit(x_train_pr,y_train)
    Rsqu_test.append(RigeModel.score(x_test_pr,y_test))
    Rsqu_train.append(RigeModel.score(x_train_pr,y_train))

#We plot the results to see for ourselves
width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(ALFA,Rsqu_test,label='validation data  ')
plt.plot(ALFA,Rsqu_train,'r',label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()