#Now we are going to use Decision Trees to determine the classes of the data in skulls.csv
import numpy as np 
import pandas 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
#These are used to create the image
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

my_data = pandas.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/skulls.csv", delimiter=",")

#We are going to get four values from my_data:
#X as the Feature Matrix (data of my_data). This is the dataset.
# Remove the column containing the target name since it doesn't contain numeric values.
# axis=1 means we are removing columns instead of rows.
X = my_data.drop(my_data.columns[[0,1]], axis=1).values
#y as the response vector (target). This is the class set.
y = my_data["epoch"]
#targetNames as the response vector names (target names). In this case each of the possible classes we have.
targetNames = my_data["epoch"].unique().tolist()
#featureNames as the feature matrix column names. In this case the labels we use for each column of data.
featureNames = list(my_data.columns.values)[2:6]

#We use the train_test_split method from before to split the dataset and class set into train and test sets.
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

#Here we instantiate the tree
skullsTree = DecisionTreeClassifier(criterion="entropy")
#Here we fit the train dataset with the train class set.
skullsTree.fit(X_trainset,y_trainset)

#We predict the classes of the test dataset.
predTree = skullsTree.predict(X_testset)

#Here we print the accuracy of the classes determined by this tree.
print("DecisionTrees's Accuracy: %r " % metrics.accuracy_score(y_testset, predTree))

#Here we write the decision tree image
dot_data = StringIO()
filename = "skulltree.png"
out=tree.export_graphviz(skullsTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)