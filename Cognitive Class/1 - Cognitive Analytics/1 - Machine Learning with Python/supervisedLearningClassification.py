import numpy as np
import pandas
from sklearn.neighbors import KNeighborsClassifier

def removeColumns(pandasArray, *column):
    return pandasArray.drop(pandasArray.columns[[column]], axis=1).values

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

my_data = pandas.read_csv("https://ibm.box.com/shared/static/u8orgfc65zmoo3i0gpt9l27un4o0cuvn.csv", delimiter=",")

X = removeColumns(my_data,0,1)
y = target(my_data,1)

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X,y)
neigh7 = KNeighborsClassifier(n_neighbors=7)
neigh7.fit(X,y)

print("Neigh's Prediction: %s" % neigh.predict(X[30].reshape(1, -1)))
print("Neigh7's Prediction: %s" % neigh7.predict(X[30].reshape(1, -1)))
print('Actual: %s' % y[30])