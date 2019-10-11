import numpy as np
from PIL import Image, ImageDraw, ImageFont

# heat map visualization for self.direction
_cmap = np.array([
			[255,	0,		0, 255],
			[255,	255,	0, 255],
			[0,		255,	0, 255],
			[255,	0,		255, 255],
			[255,	0,		0, 255]
		], dtype='i')

def _caffine(phi):
	x = phi/np.pi+0.5
	l = len(_cmap)-1

	h = x*l

	if int(h) < l:
		return np.uint8((1-h%1) * _cmap[int(h)] + (h%1) * _cmap[int(h)+1])

	return _cmap[-1]


triangle_map = np.array([
	[0,0,0,0,1],
	[0,0,1,1,1],
	[1,1,1,1,1],
	[0,0,1,1,1],
	[0,0,0,0,1]
], dtype='i')


theight = 10
twidth = 20
def draw_triangle(r, x, y):
	c = [0,0,0,255]
	tw = 0
	for i in range(-theight, theight+1):
		if i < 0:
			tw += 1
		else:
			tw -= 1

		for j in range(twidth + 1 - tw, twidth + 1):
			r[x+i,y+j] = c

stripwidth, margin = 40, 20
width, height = 150, 520

lo, hi = -np.pi/2, np.pi/2

tx = np.linspace(lo, hi, height-2*margin)

r = np.array([[[0]*4]*width]*height)

for i in range(margin, height-margin):
	for j in range(stripwidth):
		r[i,j] = _caffine(tx[i-margin])
	if (i-margin)%((height-2*margin)/4) == 0:
		draw_triangle(r,i,j)
draw_triangle(r,i,j)

a = Image.fromarray(np.uint8(r))

strings = ["-0.5", "-0.25", "0", "+0.25", "+0.5"]
k = 0

fontsize = theight+5
draw = ImageDraw.Draw(a)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", fontsize)

for i in range(margin, height-margin):
	if (i-margin)%((height-2*margin)/4) == 0:
		draw.text((stripwidth+twidth+5, i-2-fontsize/2), strings[k], (0,0,0), font=font)
		k+=1

draw.text((stripwidth+twidth+5, i-2-fontsize/2), strings[k], (0,0,0), font=font)

a.save("strip.png")
