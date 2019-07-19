nbd_mask = [
	[0,0,1,1,1,0,0],
	[0,1,1,1,1,1,0],
	[1,1,1,0,1,1,1],
	[1,1,0,1,0,1,1],
	[1,1,1,0,1,1,1],
	[0,1,1,1,1,1,0],
	[0,0,1,1,1,0,0]
]
nbd_center = (3,3)

def get_nbd(nbd_mask,nbd_center,i,j):
	nbd = []
	for i2 in range(len(nbd_mask)):
		for j2 in range(len(nbd_mask[0])):
			if nbd_mask[i2][j2] == 1:
				nbd.append((i2+i-nbd_center[0],j2+j-nbd_center[1]))
	return nbd

A = get_nbd(nbd_mask, nbd_center, 5,5)
s = ""
for i2 in range(11):
	s += "\n"
	for j2 in range(11):
		if (i2,j2) in A:
			s += "x"
		else:
			s += " "

print(s)