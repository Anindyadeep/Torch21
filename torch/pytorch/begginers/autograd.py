import torch

'''
just writting a simple
code for a particular neuron in the 
neural network
'''
x = torch.rand(5)
y = torch.zeros(3)
w = torch.rand(5,3, requires_grad=True)
b = torch.zeros(3, requires_grad=True)
z = torch.matmul(x,w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

loss.backward()
print(w.grad)
print(b.grad)

# see whether the gradient is applied to the param or not
print(w.requires_grad)

# disabling the gradient 

with torch.no_grad(): 
    # here in this block the computations will not be tracked up
    z = torch.matmul(x,w)+b

# and so after this the gradient computation for z is been turned off
print(z.requires_grad)

'''
(source pytorch documantation)
Conceptually, autograd keeps a record of data (tensors) and all executed operations
(along with the resulting new tensors) in a directed acyclic graph (DAG) consisting
of Function objects. In this DAG, leaves are the input tensors, roots are the output
tensors. By tracing this graph from roots to leaves, you can automatically compute the
gradients using the chain rule.
'''
