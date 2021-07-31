import os
import pandas as pd
from torchvision import datasets
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader

'''
This is genarally done when we get some dataset from online 
which is genarally taken from online i.e. downloaded online
so in pytorch it is very essential to organise those kinds of data
and this customImageDataset class is very much important for 
loading, manipualating, and doing other stuffs with the data
'''

class CustomImageDataset(Dataset):
	def __init__(self, annotation_file, img_dir, transform = None, target_transform = None):
		'''
		--> the annotations file will contain the labels of the image
		--> img_dir will contain the directory of the images
		--> transform will genarally lead to the transformation of the images to tensors
		'''

		self.img_labels = annotation_file
		self.img_dir = img_dir
		self.transform = transform
		self.target_transform = target_transform

	# now we will make a function that will get the length of the dataset
	def __len__(self):
		return len(self.img_labels)


'''
Now we will create a function that will return the tensors as well
and we have to return the tensors and also will make us visualize the 
things about how the data will look like
'''

class LoadDataTensors():
	def __init__(self, training_data, testing_data, batch_size, shuffle, show = False):
		self.batch_size = batch_size
		self.shuffle = shuffle

		self.train_data_loader = DataLoader(training_data, batch_size = self.batch_size, shuffle = self.shuffle)
		self.testing_data_loader = DataLoader(testing_data, batch_size = self.batch_size, shuffle = self.shuffle)

		training_features , training_labels = next(iter(self.train_data_loader))
		testing_features, testing_labels = next(iter(self.testing_data_loader))

		print(f"The size of the training data is: {training_features.size()}")
		print(f"The size of the testing data is: {testing_features.size()}")

		img = training_features[0].squeeze(0)
		label = training_labels[0]

		if show:
			plt.imshow(img)
			plt.title(label)
			plt.show()
		return (training_features, training_labels), (testing_features, testing_labels)