import torch
import torch.nn as nn           # import the whole neural net module containing all operations
import torch.optim as optim     # imports all the kinds of optimizers
import torch.nn.functional as F # imports all the kinds of activation functions

# The datasets parts
from torch.utils.data import DataLoader
import torchvision.datasets as datasets     # sample datasets provided by pytorch
import torchvision.transforms as transforms # the data augmentation part
from tqdm import tqdm

'''
First we will create a simple class NN, which is a 
neural network class, and this class will contain, a 
simple neural network architecture, that will contain -->
                            1. 2 hidden layers
                            2. One is of (input_size x 50) [weights dimension]
                            3. other is (50 x num_classes) [weights dimension]
This will do the linear operations or we can call this architecture as
the feed forward architecture as well.
'''

# create the fully connected layer

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        # note fc --> fully_connected layer
        '''
        Actually here we are defining the dimensions of 
        the weights only which will be used in
        the feed forward computations
        '''
        self._fc1 = nn.Linear(input_size, 50)
        self._fc2 = nn.Linear(50, 20)
        self._fc3 = nn.Linear(20, num_classes)
    
    def forward(self, x):
        '''
        Here we will do the feed forward computations
        the core computations will be done by pytorch, 
        just we have to define it intuitively. Also initially
        the x = input so let us assume we have an input of shape (64, 784)
        
        The computations are as follows -->
        1. (64, 784) * fc1 = (64, 784) * (784, 50) = (64, 50) [x = (64, 50)]
        2. x = (64, 50) * (50, 10) = (64, 10)
        
        where 784 is the size of the input
        and 64 is the number of example
        the x I am showing the shape, not the matrix
        '''
        x = F.relu(self._fc1(x))
        x = F.relu(self._fc2(x))
        x = self._fc3(x) 
        return x
  
# set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

# setting up the hyper-parameters
input_size = 784
num_classes = 10
learning_rate = 0.001
epochs = 3
batch_size = 64

# setting up the dataset
'''
root -->      it selects where to download, or from where the data should
              be taken from
         
train -->     it will know whether this dataset is for training or testing

download -->  it will download the dataset if True or take locally if taken

transform --> it is in the form of numpy array, which will be 
              converted to tensors for pytorch computations

'''


train_dataset = datasets.MNIST(
                        root = "dataset/", 
                        train = True, 
                        download = True,
                        transform = transforms.ToTensor())

test_dataset = datasets.MNIST(
                        root = "dataset/", 
                        train = False, 
                        download = True,
                        transform = transforms.ToTensor())

# loading the dataset with the help of the data loader

train_loader = DataLoader(
                    dataset = train_dataset,
                    batch_size = batch_size,
                    shuffle = True)

test_loader = DataLoader(
                    dataset = test_dataset,
                    batch_size = batch_size,
                    shuffle = True)

# initialising the neural network

model = NN(input_size = input_size, num_classes = num_classes).to(device)

# initialising the loss and the optimizer for the training

cross_entropy_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Train the network
'''
During training the neural network in pytorch especially
we need to follow this steps as follows -->
                1. iterate over certain number of epochs
                2. iterate over the batch size
                3. get the data and targets to the device (cude or cpu)
                4. make the shape of the data correct
                5. get the scores from the feed forward model
                6. using that scores compute the loss
                7. initialize all the grads of the optimizers to zero
                   as because we do not want to keep the gradients same for 
                   the different batches
                8. perform the gradient descent with steps
                9. and finally back propagate through out the network
'''

for epoch in range(epochs):
    for batch_no, (data, target) in enumerate(tqdm(train_loader)):
        
        data = data.to(device)
        target = target.to(device)
        
        '''
        now reshaping the data here means we want to flatten
        the data, keeping the batch size intact, and it can be done 
        easily with reshape(batch_size, -1) method, which will do it
        automatically for us
        '''
        data = data.reshape(data.shape[0], -1)
        
        # forward propagation
        logits = model(data)
        loss = cross_entropy_loss(logits, target)
        
        # backward propagation
        # initilize all the grads to zero first
        optimizer.zero_grad()
        loss.backward()
        
        # gradient descent or adam step
        optimizer.step()
        
# checking the accuracy of the model
'''
here we will do the same steps above just we will not 
calculate the gradients of the data
'''
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    
    with torch.no_grad():
        for x, y, in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0], -1)
            
            logits = model.forward(x)
            _, predictions = logits.max(1) # _ will give the index which we dont want right now
            num_correct += (predictions == y).sum()
            '''
            this actually means like this -->
            this will calculate the number of true in the tensor
            and provide the sum of that
            '''
            num_samples += predictions.size(0) # 64 here
    model.train()
    return num_correct / num_samples

# final making the model work

print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")           