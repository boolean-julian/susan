from PIL import Image
import numpy as np
import os
import sys
import multiprocessing as mp

n_proc = mp.cpu_count()
class Susan:
	# default mask with 37 neighbors per pixel
	"""
	default_mask = np.matrix([
		[0,0,1,1,1,0,0],
		[0,1,1,1,1,1,0],
		[1,1,1,1,1,1,1],
		[1,1,1,1,1,1,1],
		[1,1,1,1,1,1,1],
		[0,1,1,1,1,1,0],
		[0,0,1,1,1,0,0]
	], dtype='?')
	
	"""
	default_mask = np.matrix([
		[1,1,1],
		[1,1,1],
		[1,1,1]
	], dtype="?")
	
	
	# sets initial mask, file path and comparison function
	def __init__(self, path, mask = default_mask, compare = "exp"):
		self.load(path)
		self._set_mask(mask)

		if compare == "naive":
			self.compare = self._compare_naive
		if compare == "exp":
			self.compare = self._compare_exp


	# setters
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
			self.img = np.array(Image.open(self.path).convert("L"), dtype="i")
		except:
			print("Error: Cannot open", self.path)
			sys.exit(0)

		self.height = self.img.shape[0]
		self.width 	= self.img.shape[1]

	# get indices from mask
	def _init_nbd(self):
		self.mask_nbd = np.array([(-1,-1)]*self.nbd_size, dtype="i4")
		index = 0
		for k in range(self.mask.shape[0]):
			for l in range(self.mask.shape[1]):
				if self.mask[k,l] == 1:
					x = k-self.center[0]
					y = l-self.center[1]
					self.mask_nbd[index] = (x,y)
					index += 1

	def _compare_naive(self, a, b, t):
		if np.abs(self.img[a] - self.img[b]) <= t:
			return 1
		return 0

	def _compare_exp(self, a, b, t):
		return np.exp(-((self.img[a] - self.img[b])/t)**6)

	# n from paper
	def _nbd_compare(self, i, j, t):
		s = 0
		for r in self.mask_nbd:
			x = i+r[0]
			y = j+r[1]
			if x >= 0 and x < self.height and y >= 0 and y < self.width:
				s += self.compare((i,j), (x,y), t)
		return s
	
	def detect_edges(self, t, filename = "out.png"):
		r = self.img.copy()
		max_response = 0
		#g = .75*self.nbd_size
		
		for i in range(self.height):
			for j in range(self.width):
				r[i,j] = max(0, self.nbd_size - self._nbd_compare(i,j,t))
				
				if r[i,j] > max_response:
					max_response = r[i,j]
				
		r = r/max_response * 255
		self.save(r, filename)


	"""
	def _nbd_compare_mp(self, start, end, t, height, width, mask_nbd, nbd_size, compare, r):
		for i in range(start, end):
			print(i)
			for j in range(width):
				s = 0
				for r in mask_nbd:
					x = i+r[0]
					y = j+r[1]
					if x >= 0 and x < height and y >= 0 and y < width:
						s += compare((i,j), (x,y), t)
				r[i,j] = max(0, nbd_size - s)

	def detect_edges_mp(self, t, filename = "out.png"):
		r = np.zeros((self.height, self.width)) # shared array for mp(???)

		n_proc = mp.cpu_count() 			# number of cores
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

		jobs = [mp.Process(target = self._nbd_compare_mp, args = (chunks[i], chunks[i+1], t, self.height, self.width, self.mask_nbd, self.nbd_size, self.compare, r)) for i in range(len(chunks)-1)]
		for job in jobs:
			job.start()
		for job in jobs:
			job.join()
		
		self.save(r, filename)
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
			a = Image.fromarray(np.uint8(r))
			a.save(filename)
			print("Saved file to", filename)
		except:
			print("Error: Couldn't save", filename)
			return


	# debug
	"""
	def _nbd_check(self, i, j):
		A = self._get_nbd(i,j)
		s = ""
		for k in range(self.height):
			for l in range(self.width):
				if (k,l) in A:
					s += "0 "
				else:
					s += ". "
			s += "\n"

		t = ""
		for k in range(self.height):
			for l in range(self.width):
				if self.img[k,l] < 127:
					t += "0 "
				else:
					t += ". "
			t += "\n"

		with open("_check.txt", "w") as text_file:
			text_file.write("%s" % s)
			text_file.write("\n")
			text_file.write("%s" % t)
	"""