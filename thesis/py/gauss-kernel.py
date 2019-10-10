import numpy as np


# KERNEL
n = 5
m = 5

a = 2
b = 2

k = 1/273 * np.array([
	[1,4,7,4,1],
	[4,16,26,16,4],
	[7,26,41,26,7],
	[4,16,26,16,4],
	[1,4,7,4,1]
])


# IMAGE
width = 7
height = 7
I = np.array([
	[0,0,0,255,0,0,0],
	[0,0,0,255,0,0,0],
	[0,0,0,255,0,0,0],
	[0,0,0,255,0,0,0],
	[0,0,0,255,0,0,0],
	[0,0,0,255,0,0,0],
	[0,0,0,255,0,0,0]
])


# CONVOLUTION
def convolve2d(k, I):
	O = np.zeros((height,width))
	for x in range(height):
		for y in range(width):
			for i in range(n):
				for j in range(m):
					if x-i+a >= 0 and x-i+a < height and y-j+b >= 0 and  y-j+b < width:
						O[x,y] += k[i,j] * I[x-i+a, y-j+b]

	return O

O = convolve2d(k,I)
s = ""
for i in range(height):
	for j in range(width):
		number = int(np.round(O[i,j], 0))
		color = int(np.round(number/255*100,2))
		if j >= 1:
			s += "&"
		s += "|[fill=white!" + str(color) + "!black]|" + str(number) + "\t"
	s += "\\\\ \n"

print(s)



"""

RESULT:
0	11	43	69	43	11	0	
0	15	58	93	58	15	0	
0	16	62	100	62	16	0	
0	16	62	100	62	16	0	
0	16	62	100	62	16	0	
0	15	58	93	58	15	0	
0	11	43	69	43	11	0

"""