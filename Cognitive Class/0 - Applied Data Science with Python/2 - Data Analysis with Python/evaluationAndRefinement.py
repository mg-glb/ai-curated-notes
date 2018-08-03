#To run this file do
#1-From cmd type ipython
#2-Inside ipython CLI type: %matplotlib inline 
#3-Then type: %run ./evaluationAndRefinement.py
#Import libraries for data and numeric handling
import pandas as pd
#Import linear models
from sklearn.linear_model import LinearRegression
#Import model selection functions
from sklearn.model_selection import train_test_split
#Import cross validation score and predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

# Import clean data 
path = path='https://ibm.box.com/shared/static/q6iiqb1pd7wo8r3q28jvgsrprzezjqk3.csv'
df = pd.read_csv(path)

#First lets only use numeric data
df=df._get_numeric_data()

#TRAINING AND TESTING
#====================
x_data=df.drop('price',axis=1)
y_data=df['price']
#Split train and data sets in a 17:3 ratio. Evaluate R^2.
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)
lre=LinearRegression()
lre.fit(x_train[['horsepower']],y_train)
print('Horsepower accuracy %s' % lre.score(x_test[['horsepower']],y_test))
#Split train and data sets in a 1:9 ratio. Evaluate R^2.
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.9, random_state=0)
lre.fit(x_train1[['horsepower']],y_train1)
print('Horsepower accuracy %s' % lre.score(x_test1[['horsepower']],y_test1))
#Cross Validation partitions the set into k folds. For k times, it will perform the regression.
#Each time it will use a different fold as the test data set.
#When it fits, it will return a k-length list with the R^2 of each iteration.
Rcross=cross_val_score(lre,x_data[['horsepower']], y_data,cv=4)
print("The mean of the folds are %s and the standard deviation is: %s" % (Rcross.mean(),Rcross.std()))
#Now we can do the same with mean squared error
MSE=-1*cross_val_score(lre,x_data[['horsepower']], y_data,cv=4,scoring='neg_mean_squared_error')
print("The mean of the folds are %s and the standard deviation is: %s" % (MSE.mean(),MSE.std()))
#With cross_val_predict, we create the result that is the average of the four folds.
yhat=cross_val_predict(lre,x_data[['horsepower']], y_data,cv=4)
print(yhat.shape)