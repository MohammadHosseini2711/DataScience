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






