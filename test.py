from Susan import *
import numpy as np

if __name__ == '__main__':
	#S = Susan("examples/original/original.png")
	S = Susan("examples/me/me.png", mask = "mask37", compare = "exp_lut")
	S.detect_edges_mp(20, filename="out.png", geometric = True, nms = True)