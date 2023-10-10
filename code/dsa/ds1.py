def introduce_python_collections():
    # List data structure
    list = [1,2,3,4]
    print(f'List data structue example\n{list}')

    # Set data structure : duplicate values are not allowed
    set = {1,2,3,3}
    print(f'Set data structue example\n{set}')

    # Tuple data structure
    tuple = (1,2)
    print(f'Tuple data structue example\n{tuple}')

    tuple = ("a", "b", "c")
    print(f'Tuple data structue example\n{tuple}')

    # Dictionary data strucrure : Key:Value
    dict = {"a":1, "b":2, "book":3}
    print(f'dictionary data structure example \n {dict}')


# Matrix data structure : numpy
import numpy as np

def introduce_numpy_array():
    # imagine a row of boxes
    one_dim = np.array([1,2,3,4])
    print(f'one dimension array:\n{one_dim}')

    # imagine it as a n*m grid of boxes in a surface
    two_dim=np.array([
        [1,2,3,4],
        [6,7,8,9]
    ])
    print(f'two dimension array:\n{two_dim}')

    # imagine it as a (n*m)*k grid of boxes in a Volume
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


def numpy_anatomy():
    array=np.array([
        [1,2,3,4],
        [6,7,8,9]
    ])
    print(f'Shape: {array.shape}')
    print(f'Number of axis: {array.ndim}')
    print(f'Size: {array.size}')
    print(f'Data type: {array.dtype}')

def create_numpy_array_from_python_collection():
    array_list = np.array([1, 2, 3])
    array_tuple = np.array(((1, 2, 3), (4, 5, 6)))
    array_set = np.array({"pikachu", "snorlax", "charizard"})
    print(f'nparray from python list: {array_list}')
    print(f'nparray from tuples: {array_tuple}')
    print(f'nparray from set: {array_set}')


def create_numpy_array_from_functions():
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
    i,j =  np.indices((3,3))
    print(f'nparray all zeros:\n {zeros}')
    print(f'nparray all ones:\n {ones}')
    print(f'nparray arange: 1..10 with step 2:\n {arange}')
    print(f'nparray empty cells 2x2 matrix:\n {empty}')
    print(f'nparray linner space -1..1 number of points 10:\n {linespace}')
    print(f'nparray with shape (3,3) fill with -2:\n {full}')
    print(f'nparray indices i(row) for 2x2 tensor:\n {i}')
    print(f'nparray indices j(column) for 2x2 tensor:\n {j}')

def create_numpy_array_from_special_functions():
    # diagonal array
    diagonal = np.diag([1, 2, 3], k=0)
    # identity 
    identity = np.identity(3)
    # eye
    eye = np.eye(4, k=-1)
    # rand
    rand = np.random.rand(3,2)

    print(f'nparray diagonal array:\n {diagonal}')
    print(f'nparray idintity array:\n {identity}')
    print(f'nparray eye form array with k=(-3..3)):\n {eye}')
    print(f'nparray random array:\n {rand}')

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
