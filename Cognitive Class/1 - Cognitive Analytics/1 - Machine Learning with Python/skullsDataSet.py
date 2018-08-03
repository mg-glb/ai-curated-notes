import numpy as np
import pandas
from sklearn.neighbors import KNeighborsClassifier

# Remove the column containing the target name since it doesn't contain numeric values.
# Also remove the column that contains the row number
# axis=1 means we are removing columns instead of rows.
# Function takes in a pandas array and column numbers and returns a numpy array without
# the stated columns
def removeColumns(pandasArray, *column):
    return pandasArray.drop(pandasArray.columns[[column]], axis=1).values

#This function transforms the values from a particular column into numpy values and does two things:
#One: Returns the column with the transformed values
#Two: Returns an array with the real values, ordered by numpy value.
def targetAndtargetNames(numpyArray, targetColumnIndex):
    target_dict = dict()
    target = list()
    target_names = list()
    count = -1
    for i in range(len(my_data.values)):
        if my_data.values[i][targetColumnIndex] not in target_dict:
            count += 1
            target_dict[my_data.values[i][targetColumnIndex]] = count
        target.append(target_dict[my_data.values[i][targetColumnIndex]])
    # Since a dictionary is not ordered, we need to order it and output it to a list so the
    # target names will match the target.
    for targetName in sorted(target_dict, key=target_dict.get):
        target_names.append(targetName)
    return np.asarray(target), target_names

#Here we retrieve the .csv file from the web.
my_data = pandas.read_csv("https://ibm.box.com/shared/static/u8orgfc65zmoo3i0gpt9l27un4o0cuvn.csv", delimiter=",")
#Here we remove the index column and the date range value and we assign the remainder to new_data
new_data = removeColumns(my_data,0,1)
#As described by the method, we get the numpy-transformed array and the conversion values
target, target_names = targetAndtargetNames(my_data, 1)

X = new_data
y = target
#Here we get the K-Nearest-Neighbors method
neigh = KNeighborsClassifier(n_neighbors=1)
#Here we pair the new_data and the target in the neighbor method.
neigh.fit(X,y)
d = 1
for a,b in zip(X,y):
    c = (neigh.predict(a.reshape(1,-1))) == (b)
    if c:
        print("%s: %s" % (d,target_names[b]))
    else:
        print("%s: Could not be determined" % d)
    d += 1