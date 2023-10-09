- [Python collection types](#python-collection-types)
  - [List](#list)
  - [Set](#set)
  - [Tuple](#tuple)
  - [Dictionary](#dictionary)
- [Numpy array](#numpy-array)
  - [Anatomy of Numpy array](#anatomy-of-numpy-array)
  - [Create](#create)
    - [From python data types](#from-python-data-types)
    - [From Numpy functions](#from-numpy-functions)
    - [From special functions](#from-special-functions)
  - [indexing and slicing](#indexing-and-slicing)

## Python collection types

### List
### Set
### Tuple
### Dictionary

## Numpy array
Numpy arrays are tensors. Rank of tensors == number of indices.
- Rank = 0 : Scalar
- Rank = 1 : Array
  ![Rank1](rank1_tensor.png)
- Rank = 2 : Matrix
  ![Rank2](rank2_tensor.png)
- Rank = 3 : 3D Tensor
  ![Rank3](rank3_tensor.png)
- Rank = 4 : 4D Tensor
  ![Rank4](rank4_tensor.png)
- ....

### Anatomy of Numpy array
- Shape
- Size
- Axis
- Data type

```Python
three_dim=np.array([
    [
        [1,2,3,4],
        [6,7,8,9]
    ],
    [
        [1,2,3,4],
        [6,7,8,9]
    ],
    [
        [1,2,3,4],
        [6,7,8,9]
    ]

])
print(f'two dimension array:\n{three_dim}')


array = three_dim
print(f'Shape: {array.shape}')
print(f'Number of axis: {array.ndim}')
print(f'Size: {array.size}')
print(f'Data type: {array.dtype}')
```

### Create
#### From python data types
```Python
array_list = np.array([1, 2, 3])
array_tuple = np.array(((1, 2, 3), (4, 5, 6)))
array_set = np.array({"pikachu", "snorlax", "charizard"})
```

#### From Numpy functions
```Python
# zeros
zeros = np.zeros(5)
# ones
ones = np.ones((3, 3))
# arange
arange = np.arange(1, 10, 2)
# empty
empty =  np.empty([2, 2])
# linspace
linespace = np.linspace(-1.0, 1.0, num=10)
# full
full = np.full((3,3), -2)
# indices
indices =  np.indices((3,3))
```

#### From special functions
```Python
# diagonal array
diagonal = np.diag([1, 2, 3], k=0)
# identity 
identity = np.identity(3)
# eye
eye = np.eye(4, k=1)
# rand
rand = np.random.rand(3,2)
```

### indexing and slicing

