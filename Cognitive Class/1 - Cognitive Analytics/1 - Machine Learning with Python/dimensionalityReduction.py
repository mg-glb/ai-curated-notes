#We will first be looking at Feature Selection with VarianceThreshold.
#VarianceThreshold is a useful tool to removing features with a threshold variance.
#It is a simple and basic Feature Selection.
from sklearn.feature_selection import VarianceThreshold

#Now VarianceThreshold removes all zero-variance features by default.
sel = VarianceThreshold()
#Given the dataset below, let's try to run fit_transform function from sel on it.
dataset = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
reduced_dataset=sel.fit_transform(dataset)
#Now you should have only three features left.
print(reduced_dataset)

#Now let's instantiate another VarianceThreshold but with a threshold of 60%.
sel60 = VarianceThreshold(threshold=(0.6 * (1 - 0.6)))
reduced_dataset=sel60.fit_transform(dataset)
#Now you should have only one feature left.
print(reduced_dataset)

#Now let's look at Univariance Feature Selection
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2 
import numpy as np 
import pandas

my_data = pandas.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/skulls.csv", delimiter=",")

# Remove the column containing the target name since it doesn't contain numeric values.
# Also remove the column that contains the row number
# axis=1 means we are removing columns instead of rows.
# Function takes in a pandas array and column numbers and returns a numpy array without
# the stated columns
def removeColumns(pandasArray, *column):
    return pandasArray.drop(pandasArray.columns[[column]], axis=1).values

X = removeColumns(my_data, 0, 1)

#Now use the target function to obtain the Response Vector of my_data and store it as y
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

y = target(my_data, 1)
#Now we will use the fit_transform function with parameters X, y of SelectKBest with parameters chi2, k=3.
X_new = SelectKBest(chi2, k=3).fit_transform(X, y)

#DictVectorizer is a very simple Feature Extraction class as it can be used to convert feature arrays
# in a dict to NumPy/SciPy representations.
from sklearn.feature_extraction import DictVectorizer

#We will use the following dictionary to be converted.
a={'Day': 'Monday', 'Temperature': 18}
b={'Day': 'Tuesday', 'Temperature': 13}
c={'Day': 'Wednesday', 'Temperature': 7}
dataset = [a,b,c]

#Now create an instance of DictVectorizer called vec
vec=DictVectorizer()
x_values=vec.fit_transform(dataset).toarray()
print(x_values)
print(vec.get_feature_names())

#Now we will use PCA to represent the data we used in feature selection(X_new)
# and project it's dimensions so make sure you have completed that portion!
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition

#Create the PCA object and train it with X_new (remember that it has three features)
pca = decomposition.PCA(n_components=2)
pca.fit(X_new)
#Using PCA we transform the three features into 2. Mathematically speaking, we do a planar transformation.
PCA_X = pca.transform(X_new)

#Plot the transformed data.
fig = plt.figure(1, figsize=(10, 8))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=0, azim=0)
ax.scatter(PCA_X[:, 0], PCA_X[:, 1], c=y, cmap=plt.cm.spectral)
plt.savefig('featureReduction.png')