#Now we are going to use Random Forests to determine the classes of the data in skulls.csv
import numpy as np 
import pandas 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

#These imports are used to create the tree image
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

my_data = pandas.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/skulls.csv", delimiter=",")

X = my_data.drop(my_data.columns[[0,1]], axis=1).values
y = my_data["epoch"]
targetNames = my_data["epoch"].unique().tolist()
featureNames = list(my_data.columns.values)[2:6]

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

#Here we instantiate the Random Forest. Here we determine the number of trees we set the criterion to be
#information gain.
skullsForest = RandomForestClassifier(n_estimators=10,criterion="entropy")
#Here we train and predict with the forest
skullsForest.fit(X_trainset,y_trainset)
predForest = skullsForest.predict(X_testset)

#The accuracy of a forest is the average of the accuracy of each tree.
print("RandomForests's Accuracy: %r" % metrics.accuracy_score(y_testset, predForest))

#Now we prepare ourselves to draw the tree image.
dot_data = StringIO()
filename = "skullforest.png"
out=tree.export_graphviz(skullsForest[1],feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)