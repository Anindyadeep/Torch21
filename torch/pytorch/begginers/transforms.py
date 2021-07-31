import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

'''
ToTensor() --> transform the PIL image to tensors but on the other hand, if we want 
               to renormalize the images into the standard deviations then we have to apply
               some series of transformations, such that things will happen -->
               1. convert PIL image to tensor (done by torch.transforms.ToTensor())
               2. Now re-normalize into the standard deviations (done by torch.transforms.Normalize())
'''

series_transform = transforms.Compose(
				[transforms.ToTensor(),
				transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

num_classes = 10
target_transform = transforms.Lambda(
			lambda y: torch.zeros(num_classes, dtype = torch.float32).scatter_(dim = 0, index = torch.tensor(y), value = 1))

'''
meaning of each line of the transform -->
for all y: we take the zeros vector whose size = (number of the classes) such that 
we will change along the rows of that tensor who
value at index = y is equal to 1
'''

train_data = datasets.CIFAR10(
			root = "dataset", 
			download = False, 
			train = True, 
			transform = series_transform,
			target_transform = target_transform)

test_data  = datasets.CIFAR10(
			root = "dataset",
			download = False,
			train = False,
			transform = series_transform,
			target_transform = target_transform)

img_train, target_train = train_data[66]
img_test, target_test = test_data[66]

print(img_train.shape, target_train.shape)
print(img_test.shape, target_test.shape)

# some visualisation part and more transformations

img_train = img_train.permute(2,1,0)
img_test = img_test.permute(2,1,0)

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

figure = plt.figure(figsize=(8,8))
figure.add_subplot(1,2,1)
plt.imshow(img_train)
plt.title(labels_map[int(torch.argmax(target_train))])
plt.axis('off')

figure.add_subplot(1,2,2)
plt.imshow(img_test)
plt.title(labels_map[int(torch.argmax(target_test))])
plt.axis('off')

plt.show()