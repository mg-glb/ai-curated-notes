#Importing
import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image

raw= "bird.jpg"
im = Image.open(raw)

# uses the ITU-R 601-2 Luma transform (there are several 
# ways to convert an image to grey scale)
image_gr = im.convert("L")    
arr = np.asarray(image_gr)
imgplot = plt.imshow(arr)
imgplot.set_cmap('gray')

#Now let's convolve the set using the edge filter
kernel = np.array([[ 0, 1, 0],[ 1,-4, 1],[ 0, 1, 0],]) 
grad = signal.convolve2d(arr, kernel, mode='same', boundary='symm')

#Before we continue, let's normalize the values of the image
type(grad)
grad_biases = np.absolute(grad) + 100
grad_biases[grad_biases > 255] = 255

fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad_biases), cmap='gray')

plt.savefig("birdEdges.png")