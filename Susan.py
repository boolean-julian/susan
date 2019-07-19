from PIL import Image
import numpy as np


class Susan:
	def __init__(self, path):
		self.load(path)

		self.mask = [
			[0,0,1,1,1,0,0],
			[0,1,1,1,1,1,0],
			[1,1,1,1,1,1,1],
			[1,1,1,1,1,1,1],
			[1,1,1,1,1,1,1],
			[0,1,1,1,1,1,0],
			[0,0,1,1,1,0,0]
		]
		self.center = (3,3)


	def _set_path(self, path):
		self.path = path

	def _set_image(self, img):
		self.img = img

	def _set_mask(self, mask, center):
		self.mask = mask
		self.center = center

	def _set_mask(self, mask):
		self.mask = mask
		self.center = (len(mask)//2, len(mask[0])//2)

	def load(self, path):
		self._set_path(path)
		
		try:
			self.img = np.array(Image.open(self.path).convert("L"), dtype="i")
		except:
			print("Error: Cannot open", self.path)
			return

		self.height = self.img.shape[0]
		self.width 	= self.img.shape[1]

	def get_nbd(i,j):
		nbd = []
		for k in range(len(self.mask)):
			for l in range(len(self.mask[0])):
				if self.mask[k][l] == 1:
					nbd.append((k+i-self.center[0],l+j-self.center[1]))
		return nbd