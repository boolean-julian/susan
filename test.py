from Susan import *
import numpy as np

if __name__ == '__main__':
	#S = Susan("examples/original/original.png")
	S = Susan("me.png", mask = "mask37", compare = "exp_lut")
	S.detect_edges_mp(15, filename="geometric_lut.png", geometric=True)