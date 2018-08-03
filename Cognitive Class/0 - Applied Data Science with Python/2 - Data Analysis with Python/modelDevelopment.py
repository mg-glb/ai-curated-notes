#To run this file do
#1-From cmd type ipython
#2-Inside ipython CLI type: %matplotlib inline 
#3-Then type: %run ./modelDevelopment.py
#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Linear Regression
from sklearn.linear_model import LinearRegression
#Seaborn
import seaborn as sns
#Polynomial features
from sklearn.preprocessing import PolynomialFeatures
#Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#MSE
from sklearn.metrics import mean_squared_error
#Polynomial R^2
from sklearn.metrics import r2_score

# path of data 
path = 'https://ibm.box.com/shared/static/q6iiqb1pd7wo8r3q28jvgsrprzezjqk3.csv'
df = pd.read_csv(path)

# create the linear regression object
'''lm = LinearRegression()
#We get the values for both 'highway-mpg' and 'price'
X = df[['highway-mpg']]
Y = df['price']
#Train the linear regression model
lm.fit(X,Y)
#Create the predicted model
Yhat=lm.predict(X)'''
#Create another linear regression, this time for 'engine-size'
'''lm1 = LinearRegression()
lm1.fit(df[['engine-size']], df[['price']])
Yhat=lm1.predict(df[['engine-size']])'''

#Create the predictor array for multiple linear regression
'''lm = LinearRegression()
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
#Train the linear regression with the prices.
lm.fit(Z, df['price'])
Yhat = lm1.predict(Z)'''

# Horsepower as potential predictor variable of price
'''width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
# Peak-rpm as potential predictor variable of price
plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)'''

#Use the corr() function to determine whether 'peak-rpm' or 'highway-mpg' is a better predictor of price
'''print(df[["peak-rpm","highway-mpg","price"]].corr())'''

#Print the residual plot for 'highway-mpg' vs 'price'
'''width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df['highway-mpg'], df['price'])
plt.show()'''

#Redo the multiple linear regression and do a distribution plot
'''width = 12
height = 10
lm = LinearRegression()
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
#Train the linear regression with the prices.
lm.fit(Z, df['price'])
Y_predict = lm.predict(Z)
#Create the figure object
plt.figure(figsize=(width, height))
#The first distribution plot should have the actual values in red.
ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
#The second distribution plot should have the fitted values in blue.
sns.distplot(Y_predict, hist=False, color="b", label="Fitted Values" , ax=ax1)
#Some metadata for the plot
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')
#Show the plot
plt.show()'''

#Polynomial Regression
#The PlotPolly function will generate a regression plot using polynomial regression
def PlotPolly(model,independent_variable,dependent_variabble, Name):
    #We define the axes
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)
    #We draw the plot
    plt.plot(independent_variable,dependent_variabble,'.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ %s' % Name)
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')
    #Show the plot
    plt.show()
'''
#Define the polynomial regression variables.
x = df['highway-mpg']
y = df['price']
# Here we use a polynomial of the 3rd order (cubic) 
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
#print(p)
#Draw the plot for highway-mpg vs price
#PlotPolly(p,x,y, 'highway-mpg')
#Here we use a polynomial of the 11th order
f1 = np.polyfit(x,y,11)
p1 = np.poly1d(f1)
print(p1)
#Draw the plot for Length
PlotPolly(p1,x,y,'Length')
'''

#Use polynomial features
'''width = 12
height = 10
lm = LinearRegression()
#Create the training set Z, this array has shape (201,4)
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
#We create a PolynomialFeatures object of degree 2:
pr=PolynomialFeatures(degree=2)
#We perform the polynomial feature transform, this transformed array has shape (201,15)
Z_pr=pr.fit_transform(Z)
#Train the model with the transformed trainset.
lm.fit(Z_pr, df['price'])
#Do the necessary steps to produce the plot
Y_predict = lm.predict(Z_pr)
plt.figure(figsize=(width, height))
ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_predict, hist=False, color="b", label="Fitted Values" , ax=ax1)
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')
plt.show()'''

#Pipelines
'''width = 12
height = 10
#A pipeline consists of processing a list of tuples. Each tuple has a task string and a task object.
#The pipeline takes the first tuple, processes it, then goes to the second feeding it with the result
#of the first tuple. All up until the end of the process.
a=('scale',StandardScaler())
b=('polynomial', PolynomialFeatures(include_bias=False))
c=('model',LinearRegression())
Input=[a,b,c]
pipe=Pipeline(Input)
#Redo Z and y
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
Y = df['price']
pipe.fit(Z,Y)
ypipe=pipe.predict(Z)
#Redo the distribution plot
plt.figure(figsize=(width, height))
ax1 = sns.distplot(Y, hist=False, color="r", label="Actual Value")
sns.distplot(ypipe, hist=False, color="b", label="Pipe Values" , ax=ax1)
plt.title('Actual vs Pipe Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')
plt.show()'''

#In-sample-evaluation
#SLR
'''lm=LinearRegression()
X = df[['highway-mpg']]
Y = df['price']
lm.fit(X,Y)
#highway_mpg_fit
lm.fit(X, Y)
# Find the R^2
print(lm.score(X, Y))
Yhat=lm.predict(X)
#mean_squared_error(Y_true, Y_predict)
print(mean_squared_error(Y, Yhat))'''

#MLR
'''lm=LinearRegression()
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
Y = df['price']
# multi_fit
lm.fit(Z, Y)
# Find the R^2
print(lm.score(Z, Y))
Y_predict_multifit = lm.predict(Z)
#Determine MSE
print(mean_squared_error(df['price'], Y_predict_multifit))'''

#Polynomial
'''x = df['highway-mpg']
y = df['price']
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
r_squared = r2_score(y, p(x))
#R^2
print(r_squared)
#MSE
print(mean_squared_error(y, p(x)))'''

#Prediction and Decision making
lm=LinearRegression()
X = df[['highway-mpg']]
Y = df['price']
lm.fit(X,Y)
#Create a new input and predict it
new_input=np.arange(1,100,1).reshape(-1,1)
yhat=lm.predict(new_input)
#Plot the result
plt.plot(new_input,yhat)
plt.show()

#CONCLUSION
'''We have three models. To evaluate them with each other, we have to compare their R^2 and their MSE:

*What is a good R-squared value?: When comparing models, the model with the higher R-squared value is a better fit for the data.
*What is a good MSE?: When comparing models, the model with the smallest MSE value is a better fit for the data.

With this in mind, we go to the models:
SLR: Using Highway-mpg as Predictor Variable of Price.
R-squared: 0.49659118843391759
MSE: 3.16 x10^7
MLR: Using Horsepower, Curb-weight, Engine-size, and Highway-mpg as Predictor Variables of Price.
R-squared: 0.80896354913783497
MSE: 1.2 x10^7
Polynomial Fit: Using Highway-mpg as a Predictor Variable of Price.
R-squared: 0.6741946663906514
MSE: 2.05 x 10^7

=======================================================================
CONCLUSION: THE MODEL WITH THE BEST OVERALL PREDICTION CAPACITY IS MLR.
======================================================================='''