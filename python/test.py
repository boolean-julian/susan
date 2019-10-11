from Susan import *

if __name__ == '__main__':
	"""
	names = ["lena", "me", "original", "test", "tux"]
	
	t = 15
	for name in names:
		path = "../" + "examples/" + name + "/"
		S = Susan(path + name + ".png", mask="mask37", compare="exp_lut")
		S.detect_edges_mp(t, filename=path + str(t) + "_out", nms=True, thin=True, heatmap=True, geometric=True, corners = False)
	"""
	
	t = 1
	path = "../thesis/assets/thinout/"
	S = Susan(path + "test.png", mask ="mask37")
	S.detect_edges_mp(t, filename = path + "out", nms=True, thin=True, heatmap=True, geometric=True, corners = False)
	
	"""
	t = 15
	path = "../examples/tux/"
	S = Susan(path + "tux.png", mask ="mask37")
	S.detect_edges_mp(t, filename = path + "test", nms=True, thin=True, heatmap=True, geometric=True, corners = False)
	"""