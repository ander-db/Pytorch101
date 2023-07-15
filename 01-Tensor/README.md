# Tensors

## TORCH.TENSOR
- A torch.Tensor is a multi-dimensional matrix containing elements of a single data type.


## TENSOR ATTRIBUTES
- Each torch.Tensor has a torch.dtype, torch.device, and torch.layout.

### torch.dtype
A torch.dtype is an object that represents the data type of a torch.Tensor. PyTorch has twelve different data types.

### torch.device
A torch.device is an object representing the device on which a torch.Tensor is or will be allocated.

The torch.device contains a device type ('cpu', 'cuda' or 'mps') and optional device ordinal for the device type. If the device ordinal is not present, this object will always represent the current device for the device type, even after torch.cuda.set_device() is called; e.g., a torch.Tensor constructed with device 'cuda' is equivalent to 'cuda:X' where X is the result of torch.cuda.current_device().

### torch.layout (BETA)
A torch.layout is an object that represents the memory layout of a torch.Tensor. Currently, we support torch.strided (dense Tensors) and have beta support for torch.sparse_coo (sparse COO Tensors).

torch.strided represents dense Tensors and is the memory layout that is most commonly used. Each strided tensor has an associated torch.Storage, which holds its data. These tensors provide multi-dimensional, strided view of a storage. Strides are a list of integers: the k-th stride represents the jump in the memory necessary to go from one element to the next one in the k-th dimension of the Tensor. This concept makes it possible to perform many tensor operations efficiently.

```python
x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
x.stride()

x.t().stride()
```


### torch.memory_format

A torch.memory_format is an object representing the memory format on which a torch.Tensor is or will be allocated.

Possible values are:

- torch.contiguous_format: Tensor is or will be allocated in dense non-overlapping memory. Strides represented by values in decreasing order.
- torch.channels_last: Tensor is or will be allocated in dense non-overlapping memory. Strides represented by values in strides[0] > strides[2] > strides[3] > strides[1] == 1 aka NHWC order.
- torch.channels_last_3d: Tensor is or will be allocated in dense non-overlapping memory. Strides represented by values in strides[0] > strides[2] > strides[3] > strides[4] > strides[1] == 1 aka NDHWC order.
- torch.preserve_format: Used in functions like clone to preserve the memory format of the input tensor. If input tensor is allocated in dense non-overlapping memory, the output tensor strides will be copied from the input. Otherwise output strides will follow torch.contiguous_format


## TENSOR VIEWS

PyTorch allows a tensor to be a View of an existing tensor. View tensor shares the same underlying data with its base tensor. Supporting View avoids explicit data copy, thus allows us to do fast and memory efficient reshaping, slicing and element-wise operations.

For example, to get a view of an existing tensor t, you can call t.view(...).

```python
t = torch.rand(4, 4)
b = t.view(2, 8)
t.storage().data_ptr() == b.storage().data_ptr()  # `t` and `b` share the same underlying data.
b[0][0] = 3.14
t[0][0]
```


Since views share underlying data with its base tensor, if you edit the data in the view, it will be reflected in the base tensor as well.

Typically a PyTorch op returns a new tensor as output, e.g. add(). But in case of view ops, outputs are views of input tensors to avoid unnecessary data copy. No data movement occurs when creating a view, view tensor just changes the way it interprets the same data. Taking a view of contiguous tensor could potentially produce a non-contiguous tensor. Users should pay additional attention as contiguity might have implicit performance impact. transpose() is a common example.

```python
base = torch.tensor([[0, 1],[2, 3]])
base.is_contiguous()
t = base.transpose(0, 1)  # `t` is a view of `base`. No data movement happened here.
t.is_contiguous()
c = t.contiguous()
```
