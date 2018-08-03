#In this exercise, we will use a simple Support Vector Machine (svm) to predict the digit value encoded 
#inside a 8x8 pixel image.
from sklearn.datasets import load_digits
from sklearn import svm

#The dictionary digits contains the data set we will use to predict the values.
digits = load_digits()

#The digits.data array contains the matrices that encode each digit. It is a 1797x8x8 tensor.
X = digits.data
#The digits.target array contains the values of each digit. It is used to compare the answers. It is a 1797x1
#array
y = digits.target

#Here we instantiate the Support Vector Clustering (SVC) method from the svm.
#Parameters:
#C=Penalty parameter C of the error term
#gamma=Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. If gamma is 'auto' then 1/n_features will be used
# instead.
clf = svm.SVC(gamma=0.001, C=100)
#The fit method matches each entry of X with each entry of y
clf.fit(X,y)
#The clf.predict() method is what determines the value of the digit.
print('Prediction: %s' % clf.predict((X[-1]).reshape(1, -1)))
print('Actual: %s'% y[-1])

c = True
d=1
for a,b in zip(X,y):
    c &= (clf.predict(a.reshape(1,-1))) == (b)
    print("%s: %s"% (d,c))
    d+=1
print("All? %s" % c)