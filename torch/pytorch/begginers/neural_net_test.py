import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import numpy as np
import matplotlib.pyplot as plt
from torchvision.io import read_image

train_data = datasets.CIFAR10(root = "dataset", train = True, download = False, transform = ToTensor())
test_data = datasets.CIFAR10(root = "dataset", train = False, download = False, transform = ToTensor())


'''
Displaying images in a very good way in torch
NOTE: if the image dataset is downloaded from outside, such that there are different folder strcuture
	  then that time, we use --> "datasets.ImageFolder(root = ".//", transform = transforms)"
	  and this is how, it will work out and also shown in the example

	  train_data = datasets.ImageFolder(root = "../input/ct-head-scans",
                                 transform = ToTensor())

       print(train_data)
      ---- output ---
      Dataset ImageFolder
      Number of datapoints: 83
      Root location: ../input/ct-head-scans
      StandardTransform
      Transform: ToTensor()
'''


def display_images(data,index = 0, show_multiple = False):
	classes = {
			0: 'plane', 
			1: 'car', 
			2: 'bird', 
			3: 'cat', 
			4: 'deer', 
			5: 'dog',
	    	6: 'frog', 
	    	7: 'horse', 
	    	8: 'ship', 
	    	9: 'truck'}

	if show_multiple:
		figure = plt.figure(figsize=(6,6))
		rows, cols = 3,3
		for i in range(1, cols*rows+1):
			random_index = int(np.random.randint(len(data), size = 1))
			img, target = data[random_index]
			figure.add_subplot(rows, cols, i)
			transposed_image = img.permute(2,1,0)
			plt.imshow(transposed_image)
			plt.title(classes[target])
			plt.axis('off')
		plt.show()
	else:
		img, target = data[index]
		transposed_image = img.permute(2,1,0)
		plt.figure(figsize=(6,6))
		plt.imshow(transposed_image)
		plt.title(classes[target])
		plt.axis('off')
		plt.show()

display_images(train_data, 2, True)

'''
creating a custom image dataset class such that it will take the images 
from a directory and it will do these following functions

-->  __len__ will return the length of the dataset
-->  __get_item__ will return a specific image and the label of a dataset at a given index in a dict format
-->  __visualise__ will help to visualise either single or multiple images at once
'''


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform = None, target_transform = None, is_annotation_csv = False, annotations = None):
        self.data_dir = data_dir
        self.is_annotation_csv = is_annotation_csv
        if not is_annotation_csv:
            self.data = datasets.ImageFolder(root = self.data_dir, transform = ToTensor())
        else:
            if type(annotations) == str:
                self.annotation = pd.read_csv(annotations)
            else:
                self.annotation = annotations
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        if self.is_annotation_csv:
            return len(self.annotation)
        else:
            return len(self.data)
            
    def __getitem__(self, index):
        if self.is_annotation_csv:

            image_dir = os.path.join(self.data_dir, self.annotation.iloc[index][0])
            target = int(self.annotation.iloc[index][1])
            image = read_image(image_dir)
            if image.shape[0] == 1 or image.shape == 3:
                image = image.permute(2,1,0)
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                target = self.target_transform(target)
            sample = {"image": image, "target":target}
            return sample

        else:
            image, target = self.data[index]
            if image.shape[0] == 1 or image.shape[0] == 3:
                image = image.permute(2,1,0)
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                target = self.target_transform(target)
            sample = {"image": image, "target": target}
            return sample

