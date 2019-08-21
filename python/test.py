from Susan import *

if __name__ == '__main__':
	S = Susan("../examples/original/original.png", mask="mask37", compare="exp_lut")
	S.detect_edges_mp(15, filename="out.png", nms=True, heatmap=True)
