import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# downloading the data

training_data = datasets.FashionMNIST(
				root = "dataset/",
				train = True,
				download = True,
				transform = ToTensor())

testing_data = datasets.FashionMNIST(
				root = "dataset/",
				train = False,
				download = True,
				transform = ToTensor())


# visualising the dataset

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(9,9))
cols, rows = 3,3

for i in range(1, rows*cols+1):
	'''
	the size of (1,) will return just the full tensor datatype 
	and .item() will return the single value from that tensor
	''' 
	sample_idx = torch.randint(high = len(training_data), size=(1,)).item()
	img, target = training_data[sample_idx]
	print(img.shape)
	print(img.shape[:2], "\n")
	figure.add_subplot(rows, cols, i)
	'''
	it will parse throgh the rows and cols and make
	3 by 3 images and subplots for all i and plot iy
	'''

	plt.axis('off') # axis will be turned off 
	plt.imshow(img.squeeze(0)) # the shape of the tensor (1x28x28 will be converted to 28x28)
	plt.title(labels_map[target])
plt.show()