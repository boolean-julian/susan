import cv2
import numpy as np
import matplotlib.pyplot as plt

#path = "../examples/lena/lena.png"
#path = "../thesis/assets/principle-test/01.png"
path = "../examples/original/original.png"

img = cv2.imread(path)
edges = cv2.Canny(img,100,200)

plt.imsave(path + "-canny.png", edges, cmap="gray")