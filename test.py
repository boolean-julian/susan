from Susan import *
import numpy as np

if __name__ == '__main__':
	#S = Susan("examples/original/original.png")
	S = Susan("me.png")
	S.detect_edges_mp(10, filename="geometric.png", geometric=True)