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
  - [Vectorization and broadcasting](#vectorization-and-broadcasting)
  - [Element-wise operations](#element-wise-operations)
    - [Arithmatic operations](#arithmatic-operations)
    - [Trigonomic functions](#trigonomic-functions)
  - [Indexing and slicing](#indexing-and-slicing)

## Python collection types

### List
```Python
# List data structure
list = [1,2,3,4]
print(f'List data structue example\n{list}')
```
### Set
```Python
# Set data structure : duplicate values are not allowed
set = {1,2,3,3}
print(f'Set data structue example\n{set}')
```
### Tuple
```Python
# Tuple data structure
tuple = (1,2)
print(f'Tuple data structue example\n{tuple}')
tuple = ("a", "b", "c")
print(f'Tuple data structue example\n{tuple}')
```
### Dictionary
Fast search data structure
```Python
# Dictionary data strucrure : Key:Value
dict = {"a":1, "b":2, "book":3}
print(f'dictionary data structure example \n {dict}')
```

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

### Vectorization and broadcasting
Numpy arrays operations are way faster than python loops, because of vectorized operations.
```Python
import timeit
def vectorizations_vs_python_loops():
    x = np.random.rand(100)
    y = np.random.rand(100)
    start = timeit.default_timer()
    for i in range(0, len(x)):
        x[i] + y[i]

    time1 = timeit.default_timer() - start
    print(f'Time for addition of 2 arrays using python loop is : {time1}')

    start = timeit.default_timer()
    x+y
    time2 = timeit.default_timer() - start
    print(f'Time for addition of 2 arrays using numpy vectorization loop is : {time2}')
    print(f'Vectorization is {time1/time2} times faster than for loop')
```

![Vectorization versus loop](vectorization_broadcasting.png)
The term broadcasting describes how NumPy treats arrays with different shapes during arithmetic operations. Subject to certain constraints, the smaller array is “broadcast” across the larger array so that they have compatible shapes.

![Broadcasting](broadcasting.png)

![Broadcasting](broadcasting2.png)


### Element-wise operations
#### Arithmatic operations
```Python
def arithmatic_operations():
    a = np.arange(1, 10).reshape((3,3))
    b = np.arange(10,19).reshape((3,3))

    addition = a + b
    subtraction = a - b
    multiplication = a * b
    true_division = a / b
    floor_division = a // b
    remainder = np.remainder(a, b) 
    array_scalar = a * 2

    print(f'array1 :\n {a}')
    print(f'array1 :\n {b}')
    print(f'addition = a + b :\n {addition}')
    print(f'subtraction = a - b :\n {subtraction}')
    print(f'multiplication = a * b :\n {multiplication}')
    print(f'true_division = a / b :\n {true_division}')
    print(f'floor_division = a // b :\n {floor_division}')
    print(f'remainder = np.remainder(a, b) :\n {remainder}')
    print(f'scalar array multiplication (array1) = a * 2 :\n {array_scalar}')
```
#### Trigonomic functions
```Python
import matplotlib.pylab as plt
def trigonomic_functions():
    x = np.linspace(-4, 4, 200)
    # sin function
    sin = np.sin(x)
    # cosine function
    cos = np.cos(x)
    # tangent function
    tan = np.tan(x)
    y = np.linspace(-4, 4, 200)
    # sin hyperbolic function
    sinh = np.sinh(y)
    # cosine hyperbolic function
    cosh = np.cosh(y)
    # tangent hyperbolic function
    tanh = np.tanh(y)

    print(f'x:\n {x}')
    print(f'sin:\n {sin}')
    print(f'cos:\n {cos}')
    print(f'tan:\n {tan}')
    print(f'y:\n {y}')
    print(f'sinh:\n {sinh}')
    print(f'cosh:\n {cosh}')
    print(f'tanh:\n {tanh}')

    plt.style.use('dark_background')
    # %config InlineBackend.figure_format = 'retina' # to get high resolution images
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(x, sin)
    ax1.set_title("sin")
    ax2.plot(x, cos)
    ax2.set_title("cos")
    ax3.plot(x, tan)
    ax3.set_title("tan")
    plt.tight_layout()
    plt.show()
```
![Trigonomic plot](trigonometricplot.png)

### Indexing and slicing

