import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision import transforms
from torchvision import datasets

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform = None, target_transform = None, is_annotation_csv = False, annotations = None):
        self.data_dir = data_dir
        self.is_annotation_csv = is_annotation_csv
        if not is_annotation_csv:
            print("Loading your dataset...")
            self.data = datasets.ImageFolder(root = self.data_dir, transform = transforms.ToTensor())
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
            if image.shape[0] == 1 or image.shape[0] == 3:
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
    
    def visualise_data(self, index = 0, show_multiple = False, rows = 3, cols = 3, classes = None):
        if self.is_annotation_csv:
            if show_multiple:
                upper_bound = len(os.listdir(self.data_dir))
                rows, cols = rows, cols
                figure = plt.figure(figsize=(8,8))
                for i in range(1, rows*cols+1):
                    random_index = int(np.random.randint(upper_bound, size = 1))
                    image_dir = os.path.join(self.data_dir, self.annotation.iloc[random_index][0])
                    target = self.annotation.iloc[random_index][1]
                    image = read_image(image_dir)
                    if image.shape[0] == 1 or image.shape[0] == 3:
                        image = image.permute(2,1,0)
                    figure.add_subplot(rows, cols, i)
                    plt.imshow(image)
                    plt.axis('off')
                    if classes:
                        plt.title(classes[target])
                    else:
                        plt.title(target)
                plt.show()
            else:
                image_dir = os.path.join(self.data_dir, self.annotation.iloc[index][0])
                target = self.annotation.iloc[index][1]
                image = read_image(image_dir)
                if image.shape[0] == 1 or image.shape[0] == 3:
                    image = image.permute(2,1,0)
                plt.figure(figsize=(6,6))
                plt.imshow(image)
                plt.axis('off')
                if classes:
                    plt.title(classes[target])
                else:
                    plt.title(target)
                plt.show()
        else:
            if show_multiple:
                upper_bound = len(self.data)
                rows, cols = rows, cols
                figure = plt.figure(figsize=(8,8))
                for i in range(1, rows*cols+1):
                    random_index = int(np.random.randint(upper_bound, size = 1))
                    image, target = self.data[random_index]
                    figure.add_subplot(rows, cols, i)
                    if image.shape[0] == 1 or image.shape[0] == 3:
                        image = image.permute(2,1,0)
                    plt.imshow(image)
                    if classes:
                        plt.title(classes[target])
                    else:
                        plt.title(target)
                    plt.axis('off')
                plt.show()
            else:
                figure = plt.figure(figsize=(6,6))
                image, target = self.data[index]
                if image.shape[0] == 1 or image.shape[0] == 3:
                    image = image.permute(2,1,0)
                plt.imshow(image)
                plt.axis('off')
                if classes:
                    plt.title(classes[target])
                else:
                    plt.title(target)
                plt.show()


'''
if we want to test this file here and at the same time
when we will import it, it should not be seen here, the we 
should so this here as shown below
'''

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    my_data = CustomDataset(data_dir="./", transform=transform, target_transform=None, is_annotation_csv=False)
    
