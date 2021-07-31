import torch
import numpy as np

'''
First we initialize the divice that will actually see that
if cuda is available or not fir GPU purposes, if its not present then 
it will use cpu
'''

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ initializing a simple tensor in torch ------------------

'''
here we initialise a tensor with some numbers such that
the tensor will be computed on the evice (which is either CUDA [GPU] or
CPU) 

dtype --> it will tell what kind of the datatype the tensor
          will store, is it integer or float and also we specify
          the bits also, that of how much bits that datatype can allocate
          like 16, 32, 64 etc
          
requires_grad = is this parameter we have initialised is
                trainable or not, if trainable, then it will
                require some gradients and so back-propagation 
                otherwise if it is a non-trainable parameter
                then it will not do any of the gradient computation
'''

tensor = torch.tensor(
    [[1,2,3],
     [4,5,6],
     [7,8,9]], dtype = torch.float32, device = device, requires_grad = True)

print(
      f"Tensor\n{tensor}\nType:\n{tensor.dtype}\nShape:{tensor.shape}\n",
      f"The device in which it is: {tensor.device}\n Is it requires gradient: {tensor.requires_grad}")


# ------------------ different type of initializing tensors in pytorch ------------------

# 1. unintialized tensot having some shape
empty_tensor = torch.empty(size=(3,3))

# 2. a tensor initilized with zeros having some shape
null_tensor = torch.zeros(size=(3,3))

# 3. a tensor with all values = 1 having some shape
ones_tensor = torch.ones(size=(5,5))

# 4. an identity tensor with some shape
identity_tensor = torch.eye(3,3)

# 5. diagonal matrix
'''
This will create a diagonal matrix of 3x3
as diagonal matrix is always square shaped and we need to just 
specify what will be in the diagonal with the 
help of torch, like
if we need a diagonal matrix of ones then we will specify
torch.ones(num)
else if we need to a diagonal of consecutive integers then we will
use torch.arange() function and so on...
'''

diagonal_tensor = torch.diag(torch.arange(start = 0, end = 3, step = 1))
print(diagonal_tensor)

# 6. a tensor with random numbers
random_tensor = torch.rand(size=(5,3))

# ------------------ some more different types of tensor structure like numpy ------------------

'''
arange function --> it is like numpy arange which will return
                    an array of continuous elements having some common
                    difference out there.
                    e.g: [0,2,4,6,8] (common diff = 2)
'''                    
tensor_arange = torch.arange(start = 1, end = 10, step = 1)
print(tensor_arange)

# linspace function that will do same like arange instead it will do for decimal intervals
tensor_linspace  = torch.linspace(start = 0.1, end = 1, steps = 10)

print(tensor_linspace)

# a normally distributed tensor array (with standard deviation (sigma) = 1, mean (µ) = 0)
normal_dist_tensor = torch.empty(size = (1,5)).normal_(mean = 0, std = 1)

print(normal_dist_tensor)

# an uniform distributed tensor
uniform_dist_tensor = torch.empty(size=(1,5)).uniform_(0,1)
print(uniform_dist_tensor)

#############################################################################################
# ** NOTE: ** any operations / function which ends with _ in torch, its an inplace operation#
# like here we got uniform_ (it will convert the same empty array to a uniformly dist array)#
#############################################################################################

# type conversion in pytorch

x = torch.arange(start= 0, end = 6, step = 1)
print(x)

# type 1 from integer to boolean (convert non-zero to 1 else all 0)

int_to_bool = x.bool()
print(int_to_bool)

# type 2 from integer to int16
int_to_int16 = x.short()
print(int_to_int16.dtype)

# type 3 from integer to int64 [important]
int_to_int64 = x.long()
print(int_to_int64.dtype)

# type 4 halfing the interger of int64 to float16
int64_to_int32 = x.half()
print(int64_to_int32.dtype)

# type 5 conversion from int to float32 [important]
int_to_float32 = x.float()
print(int_to_float32.dtype)

# type 6 conversion from int to float64 [important]
int_to_float64 = x.double()
print(int_to_float64.dtype)

# -------------------------------- conversion of numpy to torch tensor and vice versa ------------------------------

numpy_array = np.random.randn(3,3)
torch_tensor = torch.from_numpy(numpy_array) # though this tensor will have round off values

print(type(numpy_array))
print(type(torch_tensor))

# conversion back
tensor_to_numpy = (torch_tensor.numpy())
print(type(tensor_to_numpy))

print(numpy_array)
print(torch_tensor)
print(tensor_to_numpy)


# ----------------------------------- tensor operations --------------------------------------------
x = torch.rand(2,2)
y = torch.rand(2,2)

# additions (elementwise)
z = x + y
print("addition of x, y\n", z)

'''
Now we want to do something like
we want inplace y such that y = (x+y)
i.e. let z = (x + y) and y = z [ this is the inplace operations]
in such cases, whatever operations it will be in pytorch
all the inplace operation will end after an _ like:
add_
dot_ 
'''
# so here doing an inplace addition

print("x\n", x)
print("\ny\n", y)
y.add_(x)
print("\ny\n", y)

# substraction in torch 
# method 1:
z = x-y
# method 2:
z = torch.sub(x,y)

print("substraction of x, y\n", z)

# so here doing an inplace substarction (elementwise)

print("x\n", x)
print("\ny\n", y)
x.sub_(y) # this is x-y so x.sub_(y), if it was y-x then it will be y.sub_(x)
print("\ny\n", x)

# multiplication in torch
# method 1:
z = x * y
print("matrix multiplication of x, y\n", z)
# method 2:
z = torch.mul(x, y)

# here the multiplication done will be elementwise
print("matrix multiplication of x, y\n", z)

# so here doing an inplace multiplication (element wise)

print("x\n", x)
print("\ny\n", y)
y.mul_(x)
print("\ny\n", y)


# power operations with tensors 

x = torch.tensor([1,2,3])
# method 1
x_pow = x.pow(2)
# method 2
print(x**2)

# simple comparisions in tensors
print(x >= 2)
print(x <= 2)

####################### matrix multiplication using pytorch #######################

x1 = torch.rand(5,2)
x2 = torch.rand(2,3)

# method 1
print((torch.mm(x1, x2)).shape)
mul = x1@x2

print(mul.shape)

# method 3
mul2 = x1.mm(x2)
print(mul2.shape)

####################### dot product in tensors #######################

# dot product
'''
Remember dot product is only considered for 1D vectors 
not nD vectors where n >=2
'''

x1 = torch.tensor([1,2,3])
x2 = torch.tensor([1,2,3])

print(torch.dot(x1,x2))

####################### divisions in tensors ####################

# true divide
'''
In true_divide, the matrix or the vectors or the tensors are
being divided element wise
'''

x1 = torch.tensor([4,6,8])
x2 = torch.tensor([2,3,4])

divided_tensor = torch.true_divide(x1, x2)
print(divided_tensor)
print(torch.true_divide(torch.rand(size=(2,2)), torch.rand(size=(2,2))))


####################### batch multiplicaton #######################

'''
In batch multiplication, it genarally refers to tenor multiplications
of 3D or more, which is similar to kind of the batch training in deep 
learning, where a certain number of batches like 32, 64, 128 etc 
are created and trained, and training actually refers to these bunch of 
the tensors operations
'''

batch_size = 32
n = 5
m = 3
p = 5

x1 = torch.rand(size=(batch_size, n, m))
x2 = torch.rand(size=(batch_size, m, p))

batch_multiplication = torch.bmm(x1, x2)
print(batch_multiplication.shape)


####################### broadcasting in pytorch #######################

'''
In broadcasting always remember this thing about dim and axis
Geomatrically -->
                dim = 0 (x-axis)
                dim = 1 (y-axis)

In matrix -->
            dim = 0 (rows)
            dim = 1 (cols)
'''

# a simple broadcasting example

x1 = torch.rand((5, 5))
x2 = torch.ones((1, 5))

# Shape of z is 5x5: How? The 1x5 vector (x2) is subtracted for each row in the 5x5 (x1)
z = (x1 - x2)
print(z.size())  
z = (x1 ** x2)
print(z.size())

# the sum of the elements in a tensor using

x = torch.tensor([1,2,3,5,5])
print(torch.sum(x, dim=0))

'''
Finding the max value and the index of the max value 
in the array
'''

####################### some morte very useful fucntion in pytorch in maths #######################

values_max, index_max_val = torch.max(x, dim = 0)
print(values_max, index_max_val)

values_min, index_min_val = torch.min(x, dim = 0)
print(values_min, index_min_val)

# abs will return a nD tensor with all the elements +ve
abs_x = torch.abs(torch.tensor([-1,2,-4,5,-6]))  
print(abs_x)

'''
NOTE: Generally in argmin or argmax function or any other
      function related to broadcasting pytorch takes the default 
      dim  = 0
'''
# if we only want to find the index of the max or the min value in the tensor
index_min_value = torch.argmin(x)
index_max_value = torch.argmax(x)

print("MIN: ", index_min_value)
print("MAX: ", index_max_value)


'''
NOTE: torch.mean() requires values in tehe tensor in dtype = float
      also note this yhing, the importance of the parameter dim comes 
      to play when the tensor is 2D or greater, like assume we have 
      a matrix of size=(3,4)
      
      [[1,2,3,4],
       [5,6,7,8],
       [1,2,3,4]]
      
      CASE 1:
          now if we say dim = 0 (then it will return a 1D array output)
          where the size will be (1,4) [4], i.e the mean will be row wise done
          i.e. in the same column, all the elements of different rows
          will do their mean
          [(1+5+1), (2+6+2), ...]
          
      CASE 2:
          and if dim = 1 , the all the operation will be column wise,
          but the same row of the matrix and 
          here it will return a 1D array of size = (1,3) [3] i.e.
          [(1+2+3+4), (5+6+7+8), (1+2+3+4)]
'''

z_row = torch.mean(torch.rand(size=(3,4)), dim = 0)
z_col = torch.mean(torch.rand(size=(3,4)), dim = 1)

print("dim = 0 , row wise", z_row.size())
print("dim = 1 , col wise", z_col.size())

# comparision of two tensors

x1 = torch.tensor([1,2,3])
x2 = torch.tensor([1,3,4])

print(torch.eq(x1, x2))

# sorting in the torch 
x = torch.tensor([1,7,2,5,8,5])
sorted_x, indices = torch.sort(x, dim=0, descending=False)

print(sorted_x)
print(indices) # the indices of the elements in the unsorted array

'''
torch.clamp() function is very much similar to the ReLU 
function i.e. in any tensor if the ith element is < 0 , it will be = 0
and  if > 0 then it will return that element only
'''
x = torch.tensor([0.1, 0.2, -0.3])
print(torch.clamp(x, min = 0))

# any and all in pytorch
x = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)  # True/False values

'''
if we ask, does this array have ANY of the value True?
and if we ask, does this array have ALL values = True?
'''

print(torch.any(x))
print(torch.all(x))

####################### tensor indexing in pytorch #######################

batch_size = 10
features = 25

x = torch.rand(size=(batch_size, features))
print(x[0].shape)

# this will be 25 coz in the first batch, we are wanting all the features

'''
Now if we want the first feature of all the examples
'''

print(x[:, 0].shape)

'''
Now assume, we want to get the 3rd example in the batch and its 1st 10 features out of 25
'''

print(x[2, 0:10].shape)

# fancy indexing
'''
Suppose we want to get the multiple values for the different indices at the same time
and in order to do that, we will define an index, that will essentially pikup the indices 
specified in the list out there
'''

x = torch.arange(10)
indices_to_pick = [2,5,6]

print(x[indices_to_pick]) # will return 2, 5, 6

'''
now suppose we want to do the same thing for a 2D array or greater dim
'''

matrix = torch.rand(size=(3,5))
rows = torch.tensor([1,0])
cols = torch.tensor([4,0])

'''
now after saying tha indices, we want, it will return a tensor of len 2 where
it will have the element at (1,4) and the element at (0,0)
'''

print(matrix)
print("\n", matrix[rows, cols])

# indexing based on a certain conditions

'''
here we will define the indexing based on condition that we will pass
and according to that very conditions the, indices will be formed and corresponding 
elements will be returned in the tensor output
'''

x = torch.rand(size=(1,10))
print(x[(x < 2) | (x > 10 )])

'''
here it will print those elements whose index are either < 2 
or whose index are > 10, only those elements will be returned holding that 
very indices. So here in this example -->

0, 1, 11, 12 .... 19 these are indices that will return and the numbers of those
indices will be returned finallly and printed
'''

print(x[x.remainder(2) == 0])  # will be [0, 2, 4, 6, 8, 10, ....], i.e. indices divisible by 2

# some more useful fucntions in pytorch
'''
torch.where() --> is a very cool function that chenges the output values according to the 
                  conditions passed through them
'''

x = torch.arange(10)
print(torch.where(x > 5, x, x*2))

'''
This will return [ 0,  2,  4,  6,  8, 10,  6,  7,  8,  9]
because, it will try to see for all x which are > 5 will return as it is
and all other will be doubled than the initial x values

(x < 5)
0 --> 0*2 = 0
1 --> 1*2 = 2
2 --> 2*2 = 4
3 --> 3*3 = 6
4 --> 4*2 = 8
5 --> 5*2 = 10

(x > 5)
6 --> 6
7 --> 7
8 --> 8
9 --> 9

and so becomes the final output ==> [ 0,  2,  4,  6,  8, 10,  6,  7,  8,  9]
'''

# in order to get the unique values in a tensor is similar to the python set function
x = torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique()  # x = [0, 1, 2, 3, 4]

# finding the number of dimension in the tensor
x = torch.rand(size=(32, 8, 8))
print(x.ndimension())

# printing the number of the elements present in the tensor -->
print(x.numel()) # 32 * 8 * 8 (counting all the element inside the array)

####################### tensor re-shaping #######################


x = torch.arange(9)

print(x.reshape(3,3))
print(x.view(3,3))

'''
view() and reshape() are very much similar to each other
in genaral but  view acts on contiguous tensors meaning if the
tensor is stored contiguously (consecutive/next to each other) in memory or not, 
whereas for reshape it doesn't matter because it will copy the
tensor to make it contiguously stored, which might come
with some performance loss.

and sometimes view sometimes can cause error in some cases, where as
reshape doesnot.
'''

# tensor concatinations
x1 = torch.rand(size=(2,5))
x2 = torch.rand(size=(2,5))

'''
NOTE: during concatination if we keep dim = 0, it will concatenate row wise, i.e
      they will be joined by rows so here the final dimension will be --> (4, 5)
      and if dim = 1, then it will be joined columnwise and the final dimension
      will be --> (2, 10)
'''

print(torch.cat((x1, x2), dim=0).shape)
print(torch.cat((x1, x2), dim=1).shape)

# flattening the tensors into 1D tensors
'''
suppose we want to flatten a tensor into 1D tensor, then the torch.view(-1)
will actually do this thing for us automaticallyin no time
'''

x = torch.rand(5,6)
print(x.view(-1).shape)

# making this more advance
'''
suppose u have a 3D tensor havning some batch size, now without changing the 
batch size, we want to flatten the tensor, without flattening the whole tensor
'''

batch_size = 64
x = torch.rand(size=(batch_size, 4, 5))
print(x.view(batch_size, -1).shape)

'''
now suppose, we want to have the tensor of shape = (64, 4, 5) and
we want to transpose this whole tensor without changing the batch size
just swap the 4 and 5 to 5 and 4
'''
x_t = x.permute(0, 2, 1)
print(x_t.shape)

'''
Here 0, 2, 1 --> means we are keeping the 0th index as it is
                                          1st index in 2nd position
                                          2nd index in 1st position
                and similarly, we can make it any kind of indices we want
                and we can shuffle the tensor according to our shape
'''
# and if we have only two dim tensor then x.t() is  enough 

# transpose of 2D tensor
x = torch.rand(size=(2,5))
print(x.t().shape)

# unsqueeze() and squeeze() function in pytorch

'''
suppose we have a tensor whose shape is [10], let's say we want to add 
an additional so we have 1x10 or we want to have shape = 10x1
in this case we apply unsqueeze() function, where:
                                            0 --> will add the 1 before making (1x10)
                                            1 --> will add the 1 after making (10x1)
'''
x = torch.arange(10)
print(x.unsqueeze(0).shape)  # 1x10
print(x.unsqueeze(1).shape)  # 10x1

# suppose we want to add one more 1 in the (1x10)
z = x.unsqueeze(0).unsqueeze(0)
print(z.shape)

# Let's say we have x which is 1x1x10 and we want to remove a dim so we have 1x10
print(z.squeeze(1).shape) # can also do .squeeze(0) both returns 1x10

------------------------------------------------------------------------ THE END ---------------------------------------------------------------------------------