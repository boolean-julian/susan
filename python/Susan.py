import numpy as np
import sys
import multiprocessing as mp
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

n_proc = mp.cpu_count()

_mask37_rad = 3.4
_mask37 = np.matrix([
		[0,0,1,1,1,0,0],
		[0,1,1,1,1,1,0],
		[1,1,1,1,1,1,1],
		[1,1,1,1,1,1,1],
		[1,1,1,1,1,1,1],
		[0,1,1,1,1,1,0],
		[0,0,1,1,1,0,0]
	], dtype='?')

_mask9_rad = 1.4
_mask9 = np.matrix([
	[1,1,1],
	[1,1,1],
	[1,1,1]
], dtype="?")

class Susan:
	# default mask with 37 neighbors per pixel
	# sets initial mask, file path and comparison function
	def __init__(self, path, mask = "mask37", compare = "exp_lut"):
		self.load(path)

		if mask == "mask37":
			self._set_mask(_mask37)
			self.mask_rad = _mask37_rad
		if mask == "mask9":
			self._set_mask(_mask9)
			self.mask_rad = _mask9_rad

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
	def load(self, path, space = "L"):
		self._set_path(path)

		try:
			#self.img = cv2.imread(path, 0)
			self.img = np.array(Image.open(self.path).convert(space), dtype="i")

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
	def _compare_naive(self, img, a, b, t):
		if np.abs(img[a] - img[b]) <= t:
			return 1
		return 0

	def _compare_exp(self, img, a, b, t):
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

				i2_intra		= 0			# second moment of usan value (vertical position)
				j2_intra		= 0			# second moment of usan value (horizontal position)
				ij_intra		= 0			# second moment of usan value (sign)

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

							i2_intra += r[0]**2 * curr
							j2_intra += r[1]**2 * curr
							ij_intra += r[0] * r[1] * curr

						else:
							usan_area += 1

				i_cog = i_cog / usan_value
				j_cog = j_cog / usan_value

				# get direction for non max suppression
				direction = 2	# 'no edge' marker
				if geometric - usan_value > 0:
					distance_from_cog = np.sqrt((i_cog - i)**2 + (j_cog - j)**2)

					# inter pixel case
					if usan_area >= diam and distance_from_cog >= 1:
						if j_cog != j:
							direction = np.arctan((i_cog - i)/(j_cog - j))
						else:
							direction = np.pi/2

					# intra pixel case
					elif i2_intra != 0:
						direction = -1 * np.sign(ij_intra) * np.arctan(j2_intra/i2_intra)

					else:
						direction = np.pi/2


				index = i*self.width+j
				self.direction[index]	= direction
				self.response[index] 	= max(0, geometric - usan_value)

	_orientations = np.pi*np.array([-0.375, -0.125, 0.125, 0.375])
	def _suppress_nonmax_mp(self, start, end):
		for i in range(start, end):
			if i >= 1 and i < self.height-1:
				for j in range(1,self.width-1):
					if self.direction[i*self.width+j] != 2:
						max_here = True
						index = i*self.width+j
						r_curr = self.response[index]

						# negative diagonal
						if self.direction[index] > self._orientations[0] and self.direction[index] <= self._orientations[1]:
							if self.response[index+1-self.width] > r_curr or self.response[index-1+self.width] > r_curr:
								max_here = False

						# vertical
						elif self.direction[index] > self._orientations[1] and self.direction[index] <= self._orientations[2]:
							if self.response[index+1] > r_curr or self.response[index-1] > r_curr:
								max_here = False

						# positive diagonal
						elif self.direction[index] > self._orientations[2] and self.direction[index] <= self._orientations[3]:
							if self.response[index+self.width+1] > r_curr or self.response[index-self.width-1] > r_curr:
								max_here = False

						# horizontal
						else:
							if self.response[index+self.width] > r_curr or self.response[index-self.width] > r_curr:
								max_here = False

						# apply nonmax suppression
						if not max_here:
							self.response[index] = 0

	def __execute_and_wait(self, jobs):
		for job in jobs:
			job.start()
		for job in jobs:
			job.join()

	def detect_edges_mp(self, t, filename = "out.png", geometric = True, nms = True, thin = True, heatmap = False, overlay = True):
		self.response 	= mp.Array('d', self.width*self.height) # shared array for final image (flat)
		self.direction 	= mp.Array('d', self.width*self.height)

		if not self._has_lut:
			self._init_lut(t)

		if geometric:
			g = .75*self.nbd_size
		else:
			g = self.nbd_size

		n_proc = mp.cpu_count()			# number of cores
		chunk_size = (self.height)//n_proc
		remainder = (self.height)%n_proc

		# find appropriate chunking
		pivot = 0
		chunks = np.uint16(np.zeros(n_proc+1))
		chunks[0] = 0
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
		self.__execute_and_wait(jobs)
		R = self.__unflatten(self.response)/max(self.response)*255
		self.save(R, "no_nms_"+filename)

		# Directional heatmap, basically gradient at the edges
		if heatmap:
			sns.heatmap(self.__unflatten(self.direction), cmap="YlGnBu")
			plt.savefig("heatmap_"+filename)

		# Non-max suppression
		if nms:
			jobs = [mp.Process(
					target = self._suppress_nonmax_mp,
					args = (chunks[i], chunks[i+1]))
					for i in range(len(chunks)-1)
			]
			self.__execute_and_wait(jobs)
			R = self.__unflatten(self.response)/max(self.response)*255
			self.save(R, filename)

		# Thinning to be added
		if thin:
			pass

		# Overlay for edge detection
		if overlay:
			O = np.array([[[0]*3]*self.width]*self.height, dtype="i")
			for i in range(self.height):
				for j in range(self.width):
					O[i,j,0] = self.img[i,j]
					O[i,j,1] = self.img[i,j]
					O[i,j,2] = self.img[i,j]

			for i in range(self.height):
				for j in range(self.width):
					if R[i,j] != 0:
						O[i,j] = [255,0,0]

			for i in range(self.height):
				for j in range(self.width):
					if R[i,j] > 110 and i < self.height-2 and j < self.width-2 and i > 2 and j > 2:
						c = 1
						O[i-2,	j-2] 	= [0,255,0]
						O[i-1,	j-2] 	= [0,255,0]
						O[i, 	j-2] 	= [0,255,0]
						O[i+1,	j-2]	= [0,255,0]
						O[i+2,	j-2]	= [0,255,0]

						O[i-2,	j-1]	= [0,255,0]
						O[i+2,	j-1]	= [0,255,0]

						O[i-2,	j]	= [0,255,0]
						O[i+2,	j]	= [0,255,0]

						O[i-2,	j+1]	= [0,255,0]
						O[i+2,	j+1] = [0,255,0]

						O[i-2,	j+2] = [0,255,0]
						O[i+2,	j+2] = [0,255,0]

						O[i-2,	j+2] 	= [0,255,0]
						O[i-1,	j+2] 	= [0,255,0]
						O[i, 	j+2] 	= [0,255,0]
						O[i+1,	j+2]	= [0,255,0]
						O[i+2,	j+2]	= [0,255,0]


			self.save(O, "overlay_"+filename)







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
