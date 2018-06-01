import numpy as np
from skimage.util import img_as_float
from skimage import io,color
import matplotlib.pyplot as plt
import argparse
import math


class Cluster:
	cluster_index = 1

	def __init__(self, h, w, l=0, a=0, b=0):
		self.update(h, w, l, a, b)
		self.pixels = []
		self.no = self.cluster_index
		Cluster.cluster_index += 1

	def update(self, h, w, l, a, b):
		self.h = h
		self.w = w
		self.l = l
		self.a = a
		self.b = b


class SLIC:
	def open_image(self,rgb):
		lab_arr = color.rgb2lab(rgb)
		return lab_arr

	def save_lab_image(self,path, lab_arr):
		rgb_arr = color.lab2rgb(lab_arr)
		# rgb_arr.dtype(np.uint8)
		# io.imsave(path, rgb_arr)

	def make_cluster(self, h, w):
		return Cluster(h, w,
					   self.data[h][w][0],
					   self.data[h][w][1],
					   self.data[h][w][2])

	def __init__(self, rgb, K, M):
		self.K = K
		self.M = M

		self.data = self.open_image(rgb)
		self.image_height = self.data.shape[0]
		self.image_width = self.data.shape[1]
		self.N = self.image_height * self.image_width
		self.S = int(math.sqrt(self.N / self.K))

		self.clusters = []
		self.label = {}
		self.dis = np.full((self.image_height, self.image_width), np.inf)

	def init_clusters(self):
		h = int(self.S / 2)
		w = int(self.S / 2)
		while h < self.image_height:
			while w < self.image_width:
				self.clusters.append(self.make_cluster(h, w))
				w += self.S
			w = int(self.S / 2)
			h += self.S

	def get_gradient(self, h, w):
		if ((w + 1) >= self.image_width):
			w = self.image_width - 2

		if ((h + 1) >= self.image_height):
			h = self.image_height - 2

		gradient = self.data[h + 1][w + 1][0] - self.data[h][w][0] + \
				   self.data[h + 1][w + 1][1] - self.data[h][w][1] + \
				   self.data[h + 1][w + 1][2] - self.data[h][w][2]
		return gradient

	def move_clusters(self):
		for cluster in self.clusters:
			cluster_gradient = self.get_gradient(cluster.h, cluster.w)
			for dh in range(-1, 2):
				for dw in range(-1, 2):
					h1 = cluster.h + dh
					w1 = cluster.w + dw
					new_gradient = self.get_gradient(h1, w1)
					if new_gradient < cluster_gradient:
						cluster.update(h1, w1, self.data[h1][w1][0], self.data[h1][w1][1], self.data[h1][w1][2])
						cluster_gradient = new_gradient

	def assignment(self):
		for cluster in self.clusters:
			for h in range(cluster.h - 2 * self.S, cluster.h + 2 * self.S):
				if h < 0 or h >= self.image_height: continue
				for w in range(cluster.w - 2 * self.S, cluster.w + 2 * self.S):
					if w < 0 or w >= self.image_width: continue
					L, A, B = self.data[h][w]
					Dc = math.sqrt(
						math.pow(L - cluster.l, 2) +
						math.pow(A - cluster.a, 2) +
						math.pow(B - cluster.b, 2))
					Ds = math.sqrt(
						math.pow(h - cluster.h, 2) +
						math.pow(w - cluster.w, 2))
					D = math.sqrt(math.pow(Dc / self.M, 2) + math.pow(Ds / self.S, 2))
					if D < self.dis[h][w]:
						if (h, w) not in self.label:
							self.label[(h, w)] = cluster
							cluster.pixels.append((h, w))
						else:
							self.label[(h, w)].pixels.remove((h, w))
							self.label[(h, w)] = cluster
							cluster.pixels.append((h, w))
						self.dis[h][w] = D

	def update_cluster(self):
		for cluster in self.clusters:
			sum_h = sum_w = number = 0
			for p in cluster.pixels:
				sum_h += p[0]
				sum_w += p[1]
				number += 1
				h1 = int(sum_h / number)
				w1 = int(sum_w / number)
				cluster.update(h1, w1, self.data[h1][w1][0], self.data[h1][w1][1], self.data[h1][w1][2])

	def save_current_image(self, name):
		image_arr = np.copy(self.data)
		count = -1
		superpixels = np.zeros(image_arr.shape[:2],dtype=int)
		for cluster in self.clusters:
			count +=1
			for p in cluster.pixels:
				image_arr[p[0]][p[1]][0] = cluster.l
				image_arr[p[0]][p[1]][1] = cluster.a
				image_arr[p[0]][p[1]][2] = cluster.b

				superpixels[p[0]][p[1]] = count
			image_arr[cluster.h][cluster.w][0] = 0
			image_arr[cluster.h][cluster.w][1] = 0
			image_arr[cluster.h][cluster.w][2] = 0

			superpixels[cluster.h][cluster.w] = count
		self.save_lab_image(name, image_arr)
		return superpixels,image_arr

	def iterate_times(self,val):
		superpixels = None

		self.init_clusters()
		self.move_clusters()
		for i in range(val):
			self.assignment()
			self.update_cluster()
			name = str(self.K)+str(i)+"saliency.jpg"
			if i==val-1:
				superpixels,temp = self.save_current_image(name)
		return superpixels


if __name__ == '__main__':
	rgb = io.imread(path)
	p = SLIC(rgb, 500, 40)
	superpixels = p.iterate_times(5)
