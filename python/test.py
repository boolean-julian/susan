from Susan import *
import numpy as np

if __name__ == '__main__':
	#S = Susan("examples/original/original.png")
	S = Susan("../examples/lena/lena.png", mask = "mask37", compare = "exp_lut")
	S.detect_edges_mp(40, filename="out.png", nms = True, heatmap = True)