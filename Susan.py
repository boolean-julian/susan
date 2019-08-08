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
			self.mask_rad = 3.4
		if mask == "mask9":
			self._set_mask(mask9)
			self.mask_rad = 1.4

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

	# make array flat
	def __flatten(self, A):
		return A.flatten()

	# turn flat array back into matrix of image size
	def __unflatten(self, A):
		uf = np.zeros((self.height, self.width))
		for i in range(self.height):
			for j in range(self.width):
				uf[i,j] = A[i*self.width + j]
		return uf

	# main function for usan
	# computes usan area, usan value and gradient
	# for one chunk of height of total size (chunkend - chunkstart) * imagewidth
	# note that (i,j) = r_0 and (x,y) = r
	def _nbd_compare_mp(self, start, end, t, geometric):
		diam = 2*self.mask_rad			# mask diameter

		for i in range(start, end):
			for j in range(self.width):
				usan_area	= 0			# number of pixels elements in usan
				usan_value 	= 0			# sum over all comparisons in usan
				
				i_cog 		= 0			# center of gravity (vertical position)
				j_cog 		= 0			# center of gravity (horizontal position)
				
				i_intra		= 0			# second moment of usan value (vertical position)
				j_intra		= 0			# second moment of usan value (horizontal position)

				# calculate center of gravity and usan value
				for r in self.mask_nbd:
					x = i+r[0]
					y = j+r[1]
					
					if x >= 0 and x < self.height and y >= 0 and y < self.width:
						curr = self.compare(self.img, (i,j), (x,y), t)
						if curr != 0:
							usan_value = usan_value + curr
							
							i_cog += x * curr
							j_cog += y * curr

							i_intra += r[0]**2 * curr
							j_intra += r[1]**2 * curr

						else:
							usan_area += 1

				i_cog = i_cog / usan_value
				j_cog = j_cog / usan_value

				# get direction for non max suppression
				distance_from_cog = np.sqrt((i_cog - i)**2 + (j_cog - j)**2)
				direction = 100
				if usan_area > diam and distance_from_cog > 1:
					if j_cog != j:
						direction = np.arctan((i_cog - i)/(j_cog - j))
					else:
						direction = np.pi
				if usan_area < diam and distance_from_cog < 1:
					direction = np.arctan(i_intra/j_intra)

				self.direction[i*self.width+j]	= direction
				self.response[i*self.width+j] 	= max(0, geometric - usan_value)

	def detect_edges_mp(self, t, filename = "out.png", geometric = False):
		self.response 	= mp.Array('d', self.width*self.height) # shared array for final image (flat)
		self.direction 	= mp.Array('d', self.width*self.height)

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

		A = self.__unflatten(self.response)
		A = A/max(self.response)*255
		self.save(A, filename)
		"""
		A = self.__unflatten(self.direction)
		A = A/max(self.response)*255
		self.save(A, "directions_" + filename)
		"""

		


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




	""" deprecated
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

		directions = np.array([[0]*self.width]*self.height)
		max_response = 1
		for i in range(self.height):
			for j in range(self.width):
				r[i,j] = max(0, g - self._nbd_compare(i,j,t))
				if r[i,j] > max_response:
					max_response = r[i,j]
		self.save(r/max_response*255, filename)
	"""