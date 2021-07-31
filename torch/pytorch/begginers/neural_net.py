import torch
from torch import nn
from torch.cuda import device_count
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

'''
before defining the neural network we have
to define a device where by all the computations 
are being send (i.e. either in CPU or in GPU according to the availibility)
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"The device we are using here is {device}")

'''
defining a neural network in pytorch where
the only thing is that, we take the torch.nn.Module and inherit
to our own class, where we use the nn.moduele propeties in order to
create some custom neural net, with its own custom properties
'''

class NeuralNetwork(nn.Module):
    def __init__(self, torch_input, num_classes):
        super(NeuralNetwork, self).__init__()
        self.input_size = torch_input
        self.num_classes = num_classes
        '''
        here we will apply the Sequential() method which genarally makes our
        work quite easier to do and apply the stacking of the neural networks
        '''
        self.flatten = nn.Flatten()
        self.linear_stack_sequential = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes),
            nn.ReLU()
        )

        '''
        now we will define the forward function that will return 
        the logits/probabilities per classes which will be further utilized 
        to compute the loss function and the for the backward propagation
        '''
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_stack_sequential(x)
        logits = x
        return logits

# using the model and printing it

x = torch.rand(1,28,28, device=device)

model = NeuralNetwork(784,10)
logits = model(x)
probabilities = nn.Softmax(logits)

print(logits)
print(probabilities)

'''
defining a training loop and a testing loop
where these four things will happen -->
   1. get the data from the DataLoader()
   2. initialise the optimizer with a zero gradient
   3. the back propagate
   4. then update the parameters usng optimizer.step()
'''