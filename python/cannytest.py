import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../examples/lena/lena.png")
edges = cv2.Canny(img,100,200)

plt.imsave("canny-out.png", edges, cmap="gray")