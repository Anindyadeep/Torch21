import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


'''
assert function --> this is actually a very
					much useful function as brcause, it used to raise some
					kind of an error if the program does some thing wrong

		for e.g:
			def avg(marks):
			    assert len(marks) != 0,"List is empty."
			    return sum(marks)/len(marks)

			mark2 = [55,88,78,90,79]
			print("Average of mark2:",avg(mark2))

			mark1 = []
			print("Average of mark1:",avg(mark1))

		---- output ----

		Average of mark2: 78.0
		AssertionError: List is empty.
'''

class rescale(object):
	# here we will be ingeriting and re-write the properties of the object
	def __init__(self, output_size):
		# The isinstance() function returns True if the specified object is of the specified type, otherwise False.
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size

	def __call__(self, sample):
		image, targets = sample["image"], sample["target"]
		if image.shape[0] == 1 or image.shape[0] == 3:
			height, width = int(image.shape[1]), int(image.shape[2])
		else:
			height, width = int(image.shape[0]), int(image.shape[1])
		# checks whether the given output_size is a integer like (256) or tuple (224,224)

		if isinstance(self.output_size, int):
			if height > width:
				new_height = (self.output_size * height)/width
				new_width =  self.output_size
			else:
				new_height = self.output_size
				new_width = (self.output_size * width)/height

		else:
			new_height, new_width = self.output_size

		img_resized = transforms.resize(image, (new_height, new_width))
		targets = targets

		new_sample = {"image": img_resized, "target":targets}
		return new_sample

class RandomCrop(object):
	# random image croppinh class in python

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size == 2)
			self.output_size = output_size

	def __call__(self, sample):
		image, targets = sample['image'], sample["traget"]
		if image.shape[0] == 1 or image.shape[0] == 3:
			height, width = image.shape[1], image.shape[2]
		else:
			height, width = image.shape[0], image.shape[1]
		new_height, new_width = self.output_size

		top = np.random.randint(0, height-new_height)
		left = np.random.randint(0, width-new_width)

		cropped_image = image[top: top+new_height, left: left+new_width]
		cropped_sample = {"image": cropped_image, "target": targets}
		return cropped_sample

'''
This is how to use them
'''
scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])