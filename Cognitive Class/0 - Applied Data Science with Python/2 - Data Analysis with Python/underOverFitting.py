#To run this file do
#1-From cmd type ipython
#2-Inside ipython CLI type: %matplotlib inline 
#3-Then type: %run ./underOverFitting.py
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

#Import Polynomial Features
from sklearn.preprocessing import PolynomialFeatures

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

#Interactive PolyPlot
def f(order,test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr=PolynomialFeatures(degree=order)
    x_train_pr=pr.fit_transform(x_train[['horsepower']])
    x_test_pr=pr.fit_transform(x_test[['horsepower']])
    poly=LinearRegression()
    poly.fit(x_train_pr,y_train)
    PollyPlot(x_train[['horsepower']],x_test[['horsepower']],y_train,y_test,poly,pr)

#UNDERFITTING OVERFITTING AND MODEL SELECTION
#============================================
x_data=df.drop('price',axis=1)
y_data=df['price']
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)
lr=LinearRegression()
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']],y_train)
#Predict using train data
yhat_train=lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
#Predict using test data
yhat_test=lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

'''#Plot the distribution plot for the trained predicted results
Title='Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution '
FileName="trainDistPlot.png"
DistributionPlot(y_train,yhat_train,"Actual Values (Train)","Predicted Values (Train)",Title,FileName)
#Plot the distribution plot for the test predicted results
Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
FileName="testDistPlot.png"
DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title,FileName)'''

#Retrain the model using 55% of the data as train data
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)
'''#Do a 5th degree Polynomial Transformation
pr=PolynomialFeatures(degree=5)
x_train_pr=pr.fit_transform(x_train[['horsepower']])
x_test_pr=pr.fit_transform(x_test[['horsepower']])
#Do the adjusted polynomial regression, fit and predict
poly=LinearRegression()
poly.fit(x_train_pr,y_train)
yhat=poly.predict(x_test_pr )
#Plot the Polynomial Plot
PollyPlot(x_train[['horsepower']],x_test[['horsepower']],y_train,y_test,poly,pr)
#Do R^2 of the train data for the polynomial transform
print("R^2 of train data: %s" % poly.score(x_train_pr, y_train))
#Do R^2 of the test data for the polynomial transform
print("R^2 of test data: %s" % poly.score(x_test_pr, y_test))
#Here we do the model evaluation for different orders of polynomial.
Rsqu_test=[]
order=[1,2,3,4]
for n in order:
    #Recrate the polynomial transform of order n
    pr=PolynomialFeatures(degree=n)
    #Create the transformed predictor train data.
    x_train_pr=pr.fit_transform(x_train[['horsepower']])
    #Create the transformed predictor test data
    x_test_pr=pr.fit_transform(x_test[['horsepower']])    
    #Train the model with the transformed data
    lr.fit(x_train_pr,y_train)
    #Determine the R^2 score for the n-th order model.
    Rsqu_test.append(lr.score(x_test_pr,y_test))
#Plot the R^2 plot
plt.plot(order,Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ')
plt.savefig('r2plot.png')
#Interact with the plot
interact(f, order=(0,6,1),test_data=(0.05,0.95,0.05))'''

