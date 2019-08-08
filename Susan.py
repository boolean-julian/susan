import numpy as np
import sys
import multiprocessing as mp
from numba import jit
from PIL import Image

n_proc = mp.cpu_count()
class Susan:
	# default mask with 37 neighbors per pixel
	# sets initial mask, file path and comparison function
	def __init__(self, path, mask = "mask37", compare = "exp_lut"):
		mask37 = np.matrix([
			[0,0,1,1,1,0,0],
			[0,1,1,1,1,1,0],
			[1,1,1,1,1,1,1],
			[1,1,1,1,1,1,1],
			[1,1,1,1,1,1,1],
			[0,1,1,1,1,1,0],
			[0,0,1,1,1,0,0]
		], dtype='?')
		
		mask9 = np.matrix([
			[1,1,1],
			[1,1,1],
			[1,1,1]
		], dtype="?")
		
		self.load(path)

		if mask == "mask37":
			self._set_mask(mask37)
		if mask == "mask9":
			self._set_mask(mask9)

		self._has_lut = True
		if compare == "naive":
			self.compare = self._compare_naive
		if compare == "exp":
			self.compare = self._compare_exp
		if compare == "exp_lut":
			self.compare = self._compare_exp_lut
			self._exp_lut = np.zeros(1024)
			self._has_lut = False

	def _set_path(self, path):
		self.path = path

	def _set_image(self, img):
		self.img = img

	def _set_mask(self, mask):
		self.mask = mask
		self.center = (self.mask.shape[0]//2, self.mask.shape[1]//2)

		self.nbd_size = mask.sum()
		if self.nbd_size <= 1:
			print("Error: Mask should have more than one item (has %d)" % self.nbd_size)
			sys.exit(0)
		
		self._init_nbd()

	# loads image into susan module
	def load(self, path):
		self._set_path(path)
		
		try:
			#self.img = cv2.imread(path, 0)
			self.img = np.array(Image.open(self.path).convert("L"), dtype="i")	

		except:
			print("Error: Cannot open", self.path)
			sys.exit(0)

		self.height = self.img.shape[0]
		self.width 	= self.img.shape[1]

	# get indices from mask
	def _init_nbd(self):
		self.mask_nbd = []
		for k in range(self.mask.shape[0]):
			for l in range(self.mask.shape[1]):
				if self.mask[k,l] == 1:
					x = k-self.center[0]
					y = l-self.center[1]
					self.mask_nbd.append((x,y))


	# compare functions
	@staticmethod
	@jit(nopython=True)
	def _compare_naive(img, a, b, t):
		if np.abs(img[a] - img[b]) <= t:
			return 1
		return 0

	@staticmethod
	@jit(nopython=True)
	def _compare_exp(img, a, b, t):
		return np.exp(-((img[a] - img[b])/t)**6)

	def _init_lut(self, t):
		for c in range(-511, 512):
			self._exp_lut[c] = np.exp(-(c/t)**6)
		self._has_lut = True

	def _compare_exp_lut(self, img, a, b, t):
		return self._exp_lut[img[a]-img[b]]
	
	# n from paper
	def _nbd_compare(self, i, j, t):
		s = 0
		for r in np.array(self.mask_nbd):
			x = i+r[0]
			y = j+r[1]
			if x >= 0 and x < self.height and y >= 0 and y < self.width:
				c = self.compare(self.img, (i,j), (x,y), t)
				s += c
		return s
	
	def detect_edges(self, t, filename = "out.png", geometric = False):
		r = self.img.copy()
		
		if not self._has_lut:
			self._init_lut(t)

		if geometric:
			g = .75*self.nbd_size
		else:
			g = self.nbd_size

		directions = np.array([[[0]*2]*self.width]*self.height)
		max_response = 1
		for i in range(self.height):
			for j in range(self.width):
				r[i,j] = max(0, g - self._nbd_compare(i,j,t))
				if r[i,j] > max_response:
					max_response = r[i,j]
		self.save(r/max_response*255, filename)

	# multiprocessing
	def __flatten(self, A):
		return A.flatten()

	def __unflatten(self, A):
		uf = np.zeros((self.height, self.width))
		for i in range(self.height):
			for j in range(self.width):
				uf[i,j] = A[i*self.width + j]
		return uf

	def _nbd_compare_mp(self, start, end, t, g):
		for i in range(start, end):
			for j in range(self.width):
				s = 0
				for r in self.mask_nbd:
					x = i+r[0]
					y = j+r[1]
					if x >= 0 and x < self.height and y >= 0 and y < self.width:
						s += self.compare(self.img, (i,j), (x,y), t)
				self.r[i*self.width+j] = max(0, g - s)

	def detect_edges_mp(self, t, filename = "out.png", geometric = False):
		self.r = mp.Array('d',self.width*self.height) # shared array for final image (flat)

		if not self._has_lut:
			self._init_lut(t)

		if geometric:
			g = .75*self.nbd_size
		else:
			g = self.nbd_size

		n_proc = mp.cpu_count()			# number of cores
		chunk_size = self.height//n_proc
		remainder = self.height%n_proc

		# find appropriate chunking
		pivot = 0
		chunks = np.uint16(np.zeros(n_proc+1))

		for i in range(n_proc):
			if remainder > 0:
				pivot += chunk_size+1
				remainder -= 1
			else:
				pivot += chunk_size
			chunks[i+1] = pivot

		jobs = [mp.Process(
				target = self._nbd_compare_mp,
				args = (chunks[i], chunks[i+1], t, g))
				for i in range(len(chunks)-1)
		]
		for job in jobs:
			job.start()
		for job in jobs:
			job.join()
		
		A = self.__unflatten(self.r)
		A = A/max(self.r)*255
		self.save(A, filename)
		


	def save(self, r, filename = "a.png"):
		"""
		Saves image referenced in the ConstantDenoiser object.
		
		Parameters
		----------
		filename: String, optional
			Path, to which the image will be saved.
			Will save to ./a.png upon not specifying a String

		"""
		try:
			#cv2.imwrite(filename, np.uint8(r))
			a = Image.fromarray(np.uint8(r))
			a.save(filename)
			
			print("Saved file to", filename)
		except:
			print("Error: Couldn't save", filename)
			return