import numpy as np
import sys
import multiprocessing as mp
from PIL import Image

np.set_printoptions(precision=3)

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

							i2_intra += (r[0]**2) * curr
							j2_intra += (r[1]**2) * curr
							ij_intra += r[0] * r[1] * curr

						else:
							usan_area += 1

				i_cog = i_cog / usan_value
				j_cog = j_cog / usan_value

				self.i_cogs[i*self.width+j] = i_cog
				self.j_cogs[i*self.width+j] = j_cog

				# get direction for non max suppression
				direction = 2	# 'no edge' marker
				if geometric - usan_value > 0:
					self.dist_from_cog[i*self.width+j] = np.sqrt((i_cog - i)**2 + (j_cog - j)**2)
					
					# inter pixel case
					if usan_area > diam and self.dist_from_cog[i*self.width+j] > 1:
						if j_cog != j:
							direction = np.arctan((i-i_cog)/(j-j_cog))
					
						else:
							direction = np.pi/2
							

					# intra pixel case
					elif i2_intra != 0:
						phi = (j2_intra/i2_intra)
						if ij_intra == 0:
							direction = np.arctan(phi)
						else:
							direction = -1 * np.sign(ij_intra) * np.arctan(phi)

					else:
						direction = np.pi/2

				index = i*self.width+j
				self.direction[index]	= direction
				self.response[index] 	= max(0, geometric - usan_value)

	# classification of direction and non max suppression
	# keep in mind that the directional values for each pixel are stored in self.direction
	# which is computed in the _nbd_compare_mp function
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


	# heat map visualization for self.direction
	_cmap = np.array([
				[255,	0,		0],
				[255,	255,	0],
				[0,		255,	0],
				[255,	0,		255],
				[255,	0,		0]
			], dtype='i')

	def _caffine(self, phi):
		x = phi/np.pi+0.5
		l = len(self._cmap)-1

		h = x*l

		if int(h) < l:
			return np.uint8((1-h%1) * self._cmap[int(h)] + (h%1) * self._cmap[int(h)+1])

		return self._cmap[-1]

	def __make_heatmap(self, filename):
		A = np.linspace(-np.pi/2,np.pi/2,100)
		for a in A:
			self._caffine(a)
		
		O = np.array([[[0]*3]*self.width]*self.height, dtype="i")
		
		for i in range(self.height):
			for j in range(self.width):
				if self.direction[i*self.width+j] == 2:
					O[i,j] = [0,0,0]
				else:
					O[i,j] = self._caffine(self.direction[i*self.width+j])
		
		self.save(O, filename + ".png")



	# thinning algorithm - this needs to be thoroughly improved.
	_direct_neighbors = np.array([
						[-1,-1],[-1, 0],[-1, 1],
						[ 0,-1],        [ 0, 1],
						[ 1,-1],[ 1, 0],[ 1, 1]
	])
	def _thinout(self, start, end):
		# maximum reach of line completion
		maxlen = 3
		for i in range(start, end):
			if i >= maxlen+1 and i < self.height-maxlen-1:
				for j in range(maxlen+1,self.width-maxlen-1):
					if self.response[i+self.width*j] > 0:
						neighbor_count = 0
						dx = []
						dy = []
						for r in self._direct_neighbors:
							x = i+r[0]
							y = j+r[1]

							# filter direct neighbors with response
							if x >= 0 and x < self.height and y >= 0 and y < self.width:
								if self.response[x+self.width*y] > 0:
									neighbor_count += 1
									dx.append(r[0])
									dy.append(r[1])



						### cases for number of neighbors ###

						# probably a false positive
						if neighbor_count == 0:
							self.response[i+self.width*j] = 0


						# try line completion if only one neighbor
						elif neighbor_count == 1:
							# create array to mark pixels for line completion. will only complete line if direction of the "other side" matches.
							inbtwn = np.zeros(maxlen+2, dtype="i")
							linecnt = 0
							for k in range(len(inbtwn)):
								inbtwn[k] = (i-k*dx[0])+self.width*(j-k*dy[0])

							# rotational slackness
							for k in range(1,maxlen+2):
								eps = np.pi/32
								try:
									if self.response[inbtwn[k]] > 0 and self.direction[inbtwn[k]] > (self.direction[inbtwn[0]]-eps) and self.direction[inbtwn[k]] < (self.direction[inbtwn[0]]+eps):
										linecnt = k
								except:
									pass

							# draw line
							for k in range(1, linecnt):
								self.response[inbtwn[k]] = 0.5*self.response[inbtwn[0]] + 0.5*self.response[inbtwn[linecnt]]


						elif neighbor_count == 2:
							# check taxicab distance between the two found neighbors
							# (this allows for very quick case by case analysis without use of tedious matching algorithms)
							# (taxicab == 1,3,4 are always ok)
							taxicab = np.absolute(dx[0]-dx[1]) + np.absolute(dy[0]-dy[1])
							if taxicab == 2:
								# horizontal and vertical lines
								if dx[0] != 0 and dy[0] != 0:
									if dx[0] == dx[1]:
										if self.response[(i+dx[0])+self.width*(j+2*dy[0])] != 0 or self.response[(i+dx[0])+self.width*(j-2*dy[0])] != 0:
											self.response[i+dx[0]+self.width*j] = self.response[i+self.width*j]
											self.response[i+self.width*j] = 0
									
									else: # dy[0] == dy[1] (since taxicab == 2)
										if self.response[(i+2*dx[0])+self.width*(j+dy[0])] != 0 or self.response[(i-2*dx[0])+self.width*(j+dy[0])] != 0:
											self.response[i+self.width*(j+dy[0])] = self.response[i+self.width*j]
											self.response[i+self.width*j] = 0	
								
								# diagonal lines
								else:
									ddx = dx[0]-dx[1]
									ddy = dy[0]-dy[1]
									if self.response[(i+dx[0]+ddx)+self.width*(j+dy[0]+ddy)] != 0 or self.response[(i+dx[1]+ddx)+self.width*(j+dy[1]+ddy)] != 0:
										self.response[i+self.width*j] = 0
						
						else:
							pass


	def _detect_corners(self, start, end, delta_g):
		# get corners from edge detection response
		for i in range(start, end):
			for j in range(self.width):
				
				# if pixel is too close to cog, it's not a corner
				if self.dist_from_cog[i*self.width+j] <= np.sqrt(2):
						continue

				# check if pixels on the way to center of gravity lie in the USAN, only then we have a corner candidate
				flag = True
				
				i_dist = int(np.round(self.i_cogs[i*self.width+j]-i,0))
				j_dist = int(np.round(self.j_cogs[i*self.width+j]-j,0))

				maxdist = max(i_dist, j_dist)
				for k in range(1, maxdist+1):
					x = int(np.round(i_dist/k,0))
					y = int(np.round(j_dist/k,0))
					
					print(k,i,j,x+i, y+j)

					if self.response[(x+i)*self.width+(y+j)] == 0:
						flag = False
				
				if flag:
					self.corners[i*self.width+j] = max(0, self.response[i*self.width+j] - delta_g)


		# suppress everything that is not a local maximum
		for i in range(start, end):
			if i >= 1 and i < self.height:
				for j in range(self.width):
					if self.corners[i*self.width+j] > 0:
						for r in self._direct_neighbors:
							x = i+r[0]
							y = j+r[1]

							if x >= 0 and x < self.height and y >= 0 and y < self.width:
								if self.corners[i*self.width+j]	<= self.corners[x*self.width+y]:
									self.corners[i*self.width+j] = 0
									break
						






	def _overlay(self, filename, corners = False):
		O = np.array([[[0]*3]*self.width]*self.height, dtype="i")
		for i in range(self.height):
			for j in range(self.width):
				O[i,j,0] = self.img[i,j]
				O[i,j,1] = self.img[i,j]
				O[i,j,2] = self.img[i,j]

		for i in range(self.height):
			for j in range(self.width):
				if self.response[i*self.width+j] != 0:
					O[i,j] = [(self.response[i*self.width+j]+2*255)/3,0,0]

		# Overlay for corner detection (to be fixed)
		if corners:
			for i in range(self.height):
				for j in range(self.width):
					v = self.corners[i*self.width+j]
					if v > 0 and i < self.height-2 and j < self.width-2 and i > 2 and j > 2:
						color = [0,255,0]
						O[i-2,	j-2] 	= color
						O[i-1,	j-2] 	= color
						O[i, 	j-2] 	= color
						O[i+1,	j-2]	= color
						O[i+2,	j-2]	= color

						O[i-2,	j-1]	= color
						O[i+2,	j-1]	= color

						O[i-2,	j]		= color
						O[i+2,	j]		= color

						O[i-2,	j+1]	= color
						O[i+2,	j+1] 	= color

						O[i-2,	j+2] 	= color
						O[i+2,	j+2] 	= color

						O[i-2,	j+2] 	= color
						O[i-1,	j+2] 	= color
						O[i, 	j+2] 	= color
						O[i+1,	j+2]	= color
						O[i+2,	j+2]	= color

		self.save(O, filename + ".png")
		

	def __execute_and_wait(self, jobs):
		for job in jobs:
			job.start()
		for job in jobs:
			job.join()

	def __execute_and_save(self, filename, jobs, chunks):
		self.__execute_and_wait(jobs)
		R = self.__unflatten(self.response)/max(self.response)*255
		self.save(R, filename+".png")




	def detect_edges_mp(self, t, filename, geometric = True, nms = True, thin = False, heatmap = True, overlay = True, corners = False):
		self.response 		= mp.Array('d', self.width*self.height)
		self.direction 		= mp.Array('d', self.width*self.height)
		
		self.corners 		= mp.Array('d', self.width*self.height)
		
		self.i_cogs 		= mp.Array('d', self.width*self.height)
		self.j_cogs		 	= mp.Array('d', self.width*self.height)
		self.dist_from_cog 	= mp.Array('d', self.width*self.height)

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

		# No non-max suppression
		self.__execute_and_save(filename+"_raw", jobs, chunks)

		# Directional heatmap, basically direction of gradient of the edges
		if heatmap:
			A = self.__make_heatmap(filename+"_heat")

		# Non-max suppression
		if nms:
			jobs = [mp.Process(
					target = self._suppress_nonmax_mp,
					args = (chunks[i], chunks[i+1]))
					for i in range(len(chunks)-1)
			]
			self.__execute_and_save(filename+"_nonmax_supp", jobs, chunks)

		# Thinning (not done yet)
		if thin:
			jobs = [mp.Process(
					target = self._thinout,
					args = (chunks[i], chunks[i+1]))
					for i in range(len(chunks)-1)
			]
			self.__execute_and_save(filename+"_thinned", jobs, chunks)

		# SUSAN corner detection
		if corners:
			delta_g = 0.25 * self.nbd_size
			jobs = [mp.Process(
					target = self._detect_corners,
					args = (chunks[i], chunks[i+1], delta_g))
					for i in range(len(chunks)-1)
			]
			self.__execute_and_wait(jobs)

		# Overlay for edge detection
		if overlay:
			self._overlay(filename+"_overlay", corners)





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