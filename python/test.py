from Susan import *

if __name__ == '__main__':
	S = Susan("../examples/tux/tux.png", mask="mask37", compare="exp_lut")
	S.detect_edges_mp(25, filename="out", nms=True, thin=True, heatmap = True)