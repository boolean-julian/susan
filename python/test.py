from Susan import *

if __name__ == '__main__':
	S = Susan("../examples/lena/lena.png", mask="mask37", compare="exp_lut")
	S.detect_edges_mp(15, filename="out", nms=True, thin=True, heatmap=True, geometric=True, corners = False)