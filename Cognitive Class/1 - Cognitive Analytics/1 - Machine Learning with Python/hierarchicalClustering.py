import numpy as np 
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets.samples_generator import make_blobs

#Make the blobs
X2, y2 = make_blobs(n_samples=50, centers=[[4,4], [-2, -1], [1, 1], [10,4]], cluster_std=0.9)
#Create the model and train it
agglom = AgglomerativeClustering(n_clusters = 4, linkage = 'average')
agglom.fit(X2,y2)
# Create a minimum and maximum range of X2.
x_min, x_max = np.min(X2, axis=0), np.max(X2, axis=0)
# Get the average distance for X2.
X2 = (X2 - x_min) / (x_max - x_min)
#Create the distance matrix
dist_matrix = distance_matrix(X2,X2)
#Create the training data
Z = hierarchy.linkage(dist_matrix, 'complete')
#Create the dendogram
dendro = hierarchy.dendrogram(Z)

# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(6,4))
# These two lines of code are used to scale the data points down,
# Or else the data points will be scattered very far apart.
# Create a minimum and maximum range of X2.
x_min, x_max = np.min(X2, axis=0), np.max(X2, axis=0)
# Get the average distance for X2.
X2 = (X2 - x_min) / (x_max - x_min)
# This loop displays all of the datapoints.
for i in range(X2.shape[0]):
    # Replace the data points with their respective cluster value 
    # (ex. 0) and is color coded with a colormap (plt.cm.spectral)
    plt.text(X2[i, 0], X2[i, 1], str(y2[i]),
             color=plt.cm.spectral(agglom.labels_[i] / 10.),
             fontdict={'weight': 'bold', 'size': 9})
# Remove the x ticks, y ticks, x and y axis
plt.xticks([])
plt.yticks([])
plt.axis('off')
# Display the plot
plt.savefig('hierarchical.png')
#plt.show()