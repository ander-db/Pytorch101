###################################
########## TENSOR BASICS ##########
###################################

import torch

# Create an empty tensor. Values are not initialized yet.
x = torch.empty(5, 3)
print("Empty tensor: ", x)

# Create a randomly initialized tensor.
x = torch.rand(5, 3)
print("Random tensor: ", x)

# Create a tensor filled with zeros and of dtype long.
x = torch.zeros(5, 3, dtype=torch.long)
print("Zeros tensor: ", x)

# Create a tensor of ones.
x = torch.ones(5, 3, dtype=torch.long)
print("Ones tensor: ", x)


# Add two tensors.
x = torch.rand(5, 3)
y = torch.rand(5, 3)
print("x: ", x)
print("y: ", y)

print("x + y: ", x + y)
print("x - y: ", x - y)
print("x * y: ", x * y)
print("x / y: ", x / y)


# Add two tensors using torch.add().
print("torch.add(x, y): ", torch.add(x, y))

# Add two tensors and store the result in a tensor.
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print("torch.add(x, y, out=result): ", result)

# Add two tensors in-place.
y.add_(x)
print("y.add_(x): ", y)



# Get the size of a tensor.
print("x.size(): ", x.size())
print("x.shape: ", x.shape)


# Reshape a tensor.
x = torch.rand(4, 5)
print('x: ', x)
print('x.view(20): ', x.view(20))
print('x.view(-1): ', x.view(-1))
print('x.view(2, 10): ', x.view(2, 10))
print('x.view(2, -1): ', x.view(2, -1))
print('x.view(-1, 2): ', x.view(-1, 2))


# Tensor from numpy array.
import numpy as np

a = np.ones(5)
print("a: ", a)
b = torch.from_numpy(a)
print("b: ", b)
print("a[0] = 2")
a[0] = 2
print("a: ", a)
print("b: ", b)


# Cuda
if torch.cuda.is_available():
    print("Cuda is available.")
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    z = z.to("cpu")

# Autograd
x = torch.ones(2, 2, requires_grad=True)
print("x requires grad: ", x)



