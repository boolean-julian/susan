import numpy as np
from PIL import Image
import sys

paths = []
i = 2

mode = sys.argv[1]

cond = True
while cond:
	try:
		paths.append(sys.argv[i])
	except:
		cond = False
	i += 1

np.set_printoptions(precision = 3)

def get_string(O):
	height = O.shape[0]
	width = O.shape[1]

	s = ""
	for i in range(height):
		for j in range(width):
			number = int(np.round(O[i,j], 0))
			s += str(number) + "\t"
		s += "\n"
	return s

def tikz_string(O):
	height = O.shape[0]
	width = O.shape[1]

	s = ""
	for i in range(height):
		for j in range(width):
			number = int(np.round(O[i,j], 0))
			color = int(np.round(number/255*100,2))
			
			if j >= 1:
				s += "&"

			s += "|[fill=white!" + str(color) + "!black"
			if number >= 160:
				s += ", text=black"

			s += "]|" + str(number) + "\t"
		s += "\\\\ \n"
	return s


def nonum_string(O):
	height = O.shape[0]
	width = O.shape[1]

	s = ""
	for i in range(height):
		for j in range(width):
			number = int(np.round(O[i,j], 0))
			color = int(np.round(number/255*100,2))
			
			if j >= 1:
				s += "&"

			s += "|[fill=white!" + str(color) + "!black"
			if number >= 160:
				s += ", text=black"

			s += "]|\\phantom{H}\t"
		s += "\\\\ \n"
	return s

T = np.array([[0,1,2,2,1,0],[0,0,2,2,0,0]], dtype="i")*255*0.5
print(tikz_string(T))


for k in range(2, i-1):
	print("Opening image " + str(k-1) + " (" + paths[k-2] + ")")	
	print(paths[k-2])
	try:
		img = np.array(Image.open(paths[k-2]).convert("L"), dtype="i")
	except:
		print("Failed to open image " + str(k-1))
		print("Continuing...\n")
		continue

	if mode == "string":
		print(get_string(img))

	if mode == "tikz":
		print(tikz_string(img))

	if mode == "nonum":
		print(nonum_string(img))
	
	print("\n\n")

print("Done!")