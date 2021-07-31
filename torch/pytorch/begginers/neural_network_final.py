import torch
from torch import nn
from torch.nn import parameter
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.transforms import Lambda

num_classes = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

########################################### transformation part ###########################################

training_data_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.RandomRotation(degrees=180)
                    ])

testing_data_transform = transforms.ToTensor()
target_transform = transforms.Lambda(
                lambda y: torch.zeros(num_classes, dtype=torch.float32).scatter_(dim=0, index = torch.tensor(y), value = 1))

########################################### featching and loading data part ###########################################

training_data = datasets.CIFAR10(
                        root = "dataset/", 
                        train = True, 
                        download = True,
                        transform = training_data_transform)
                        #target_transform = target_transform) # the reason is pytorch does't support a vector in CrossEntropy() loss

testing_data =  datasets.CIFAR10(
                        root = "dataset/", 
                        train = False, 
                        download = True,
                        transform=testing_data_transform)
                        #target_transform=target_transform)

train_data_loader = DataLoader(
                        dataset=training_data, 
                        batch_size=64, 
                        shuffle=True)

test_data_loader =  DataLoader(
                        dataset=testing_data, 
                        batch_size=64, 
                        shuffle=True)

########################################### defining the neural network ###########################################

class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.flatten = nn.Flatten()
        self.linear_sequential_layers = nn.Sequential(
                        nn.Linear(self.input_size,512),
                        nn.ReLU(),
                        nn.Linear(512,128),
                        nn.ReLU(),
                        nn.Linear(128,64),
                        nn.ReLU(),
                        nn.Linear(64, self.num_classes))
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_sequential_layers(x)
        return logits
    

model = NeuralNet(3072, 10)
########################################### defining the hyperparameters ###########################################

learning_rate = 1e-3
epochs = 10

# defining the loss function
loss_function = nn.CrossEntropyLoss()

# optimizer initialisation
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

'''
enumerate() function in python

my_list = [111,222,333,444]
enumerate_obj = enumerate(my_list,1000)

for obj in enumerate_obj:
    print(obj)

output -->
(1000, 111)
(1001, 222)
(1002, 333)
(1003, 444)
'''

########################################### defining the train and test loop ###########################################

def train_loop(dataloader, model, loss_function, optimizer):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        correct = 0
        # NOTE: the batch is just 0,1,2,..... and so on...
        # prediction of the model 
        prediction = model(X)
        # define the loss
        loss = loss_function(prediction, y)

        # initialize the optimizer grads to zero
        optimizer.zero_grad()
        # autograding the whole backpropagation process
        loss.backward()
        # optimize at every step by the optimizer
        optimizer.step()

        if batch % 100 == 0:
            '''
            here we get the current loss and also 
            show them as the number of examples done ... in the training (per epochs)
            '''
            loss, current_num_examples = loss.item(), batch*len(X)
            correct += (prediction.argmax(1) == y).type(torch.float32).sum()

            print(f"LOSS IN TRAINING: {loss} after {current_num_examples}/{size}")
            print(f"ACC IN TRAINING: {(correct/size)}")

def validation_loop(dataloader, model, loss_function):
    size = len(dataloader.dataset)
    test_loss, test_correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            predictions = model(X)
            test_loss += loss_function(predictions, y).item()
            test_correct += (predictions.argmax(1) == y).type(torch.float32).sum()

            print(f"LOSS IN VALIDATION: {test_loss}")
            print(f"ACC IN VALIDATION: {(test_correct/size)*100}\n")


########################################### main training and validation loops ###########################################

epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train_loop(train_data_loader,model, loss_function, optimizer)
    validation_loop(test_data_loader, model, loss_function)
print("done...")

########################################### saving and loading the model ###########################################
torch.save(model.state_dict(), "data/model.pth")
print("Saved PyTorch Model State to model.pth")

'''
after saving the model, we can import the class of the Neural Network and set the object
and call the parameters of the neural network

model = NeuralNetwork()
model.load_state_dict(torch.load("data/model.pth"))
'''