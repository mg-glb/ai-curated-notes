#This script demonstrates the result of using K-Nearest-Neighbors with different values for K
#We will use the values of the skulls.csv file as training and testing
import numpy as np 
import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

my_data = pandas.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/skulls.csv", delimiter=",")

#As in the previous examples, we use this function to get the array that contains the epochs - the classes in
#our model
def target(numpyArray, targetColumnIndex):
    target_dict = dict()
    target = list()
    count = -1
    for i in range(len(my_data.values)):
        if my_data.values[i][targetColumnIndex] not in target_dict:
            count += 1
            target_dict[my_data.values[i][targetColumnIndex]] = count
        target.append(target_dict[my_data.values[i][targetColumnIndex]])
    return np.asarray(target)

#Also, we get the table containing the values of the my_data matrix. In this case, we are going to remove the
#header line and the two first columns.
def removeColumns(pandasArray, *column):
    return pandasArray.drop(pandasArray.columns[[column]], axis=1).values

X = removeColumns(my_data,0,1)
y = target(my_data,1)

#The train_test_split function takes four arguments:
#1-The data set
#2-The class set
#3 The test_size is the proportion we are going to use as test data.
#4 The random_state is the seed we are going to use to randomize the selection of data.
#And it returns four values:
#1-The data trainset is the part of the data set that is going to be used to learn.
#2-The data test set is the part of the data set that is going to be used to predict.
#3-The class trainset is the part of the class set that is going to be used to learn.
#4-The class test set is the part of the class set that is going to be used to evaluate at the end.
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=7)

#Each one of these represents a KNN Classifier. The first is a KNN with K=1, the second has K=23 and the last
#has K=90
neigh =   KNeighborsClassifier(n_neighbors=1)
neigh23 = KNeighborsClassifier(n_neighbors=23)
neigh90 = KNeighborsClassifier(n_neighbors=90)

#We give the data and class trainsets to each of the classifiers.
neigh.fit(X_trainset,y_trainset)
neigh23.fit(X_trainset,y_trainset)
neigh90.fit(X_trainset,y_trainset)

#We feed the classifiers with the test data. It should return the predicted classes.
pred = neigh.predict(X_testset)
pred23 = neigh23.predict(X_testset)
pred90 = neigh90.predict(X_testset)

#We use the metrics class to call the accuracy_score method. In this case it will return the percentage of
#hits we made with the classifiers.
print("Neigh's Accuracy: %s" % metrics.accuracy_score(y_testset, pred))
print("Neigh23's Accuracy: %s" % metrics.accuracy_score(y_testset, pred23))
print("Neigh90's Accuracy: %s" % metrics.accuracy_score(y_testset, pred90))
