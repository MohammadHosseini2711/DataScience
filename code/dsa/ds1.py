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


def rounding_functions():
    decimals = np.linspace(0.11111111, 0.99999999, 10)

    # rounding
    around = np.around(decimals, 3)
    # rounding
    round = np.round(decimals, 3)
    # rounding to integer
    rint = np.rint(decimals)
    # rounding integer towards zero
    fix = np.fix(decimals)
    # round to the floor
    floor = np.floor(decimals)
    # round to the ceiling
    ceil = np.ceil(decimals)

    print(f'around: {around}')
    print(f'round: {round}')
    print(f'rint: {rint}')
    print(f'fix: {fix}')
    print(f'floor: {floor}')
    print(f'ceil: {ceil}')


def exponent_logarithm():
    x = np.array([0.1, 1, np.e, np.pi])

    # exponent
    exp = np.exp(x)
    # exponent(x) -1
    expm1 = np.expm1(x)
    # 2^P
    exp2 = np.exp2(x)
    # natural log
    log = np.log(x)
    # log base 10
    log10 = np.log10(x)
    # log base 2
    log2 = np.log2(x)

    print(f'exp: {exp}')
    print(f'expm1: {expm1}')
    print(f'exp2: {exp2}')
    print(f'log: {log}')
    print(f'log10: {log10}')
    print(f'log2: {log2}')

    plt.style.use('dark_background')
    # %config InlineBackend.figure_format = 'retina' # to get high resolution images
    fig, (plot1, plot2, plot3) = plt.subplots(3, 1)
    plot1.plot(x, exp)
    plot1.set_title("exp")
    plot2.plot(x, log)
    plot2.set_title("log")
    plot3.plot(x, exp2)
    plot3.set_title("exp2")
    plt.tight_layout()
    plt.show()
    

def miscellaneous_functions():
    array_1 = np.arange(-9,9, 2)
    array_2 = np.arange(-9,9, 2).reshape((3,3))

    # sum over
    sum_1, sum_2, sum_3 = np.sum(array_1), np.sum(array_2, axis=0), np.sum(array_2, axis=1) 
    # take product
    prod_1, prod_2, prod_3 = np.prod(array_1), np.prod(array_2, axis=0), np.prod(array_2, axis=1)
    # cumulative sum
    cumsum_1, cumsum_2, cumsum_3 = np.cumsum(array_1), np.cumsum(array_2, axis=0), np.cumsum(array_2, axis=1)
    # clip values
    clip_1, clip_2 = np.clip(array_1, 2, 8), np.clip(array_2, 2, 8)
    # take absolute
    absolute_1, absolute_2 = np.absolute(array_1), np.absolute(array_2) 
    # take square root
    sqrt_1, sqrt_2 = np.sqrt(np.absolute(array_1)), np.sqrt(np.absolute(array_2)) 
    # take the square power
    square_1, square_2 =  np.square(array_1), np.square(array_2)
    # sign function
    sign_1, sign_2 = np.sign(array_1), np.sign(array_2)
    # n power
    power = np.power(np.absolute(array_1), np.absolute(array_1))


def indexing_and_slicing_one_dimension_array():
    array_one = np.arange(20,30)
    print(f'Array one: {array_one}')
    print(f'Array one dimensions: {array_one.ndim}, shape:{array_one.shape}')
    print(f'Select element at position [0]: {array_one[0]}')
    print(f'Select element at position [5]: {array_one[5]}')
    print(f'Select element at position [9]: {array_one[9]}')
    print(f'Select element at position [-5]: {array_one[-5]}')
    print(f'Select element at position [-1]: {array_one[-1]}')
    # Slicing in one dimensional array
    print(f'Elements from position [0] to position [3]: {array_one[0:3]}')
    print(f'Elements from position [5] to position [9]: {array_one[5:9]}')
    print(f'Elements from position [-9] to position [-5]: {array_one[-9:-5]}')
    print(f'Elements from position [-3] to position [-1]: {array_one[-3:-1]}')
    print(f'Elements from position [3] to position [-1]: {array_one[3:-1]}')
    # slice with stride
    print(f'Slice from position [0] to position [6] with stride [1]: {array_one[0:6:1]}')
    print(f'Slice from position [0] to position [6] with stride [2]: {array_one[0:6:2]}')
    print(f'Slice from position [-6] to position [-1] with stride [3]: {array_one[-6:-1:3]}')


def indexing_and_slicing_multi_dimension_array():
    array_two = np.arange(1,10).reshape((3,3))
    array_three = np.arange(1,9).reshape((2,2,2))
    print(f'Array two dimensions/axes: \n{array_two}\n')
    print(f'Array three dimensions/axes: \n{array_three}\n')
    print(f'Array two dimensions: {array_two.ndim}, shape:{array_two.shape}')
    print(f'Array three dimensions: {array_three.ndim}, shape:{array_three.shape}')
    # indexing: method 1
    print(f'Element at position 1 in first axis (rows) and position 1 in second axis (cols): {array_two[1,1]}')
    print(f'Element at position 0 in first axis (rows) and position 2 in second axis (cols): {array_two[0,2]}')
    print(f'Element at position -1 in first axis (rows) and position -3 in second axis (cols): {array_two[-1,-3]}')
    # indexing method 2
    print(f'Element at position 1 in first axis (rows) and position 1 in second axis (cols): {array_two[1][1]}')
    print(f'Element at position 0 in first axis (rows) and position 2 in second axis (cols): {array_two[0][2]}')
    print(f'Element at position -1 in first axis (rows) and position -3 in second axis (cols): {array_two[-1][-3]}')
    # slicing
    print(f'All elements at position 0 from first axis (all elements from first row): \n{array_two[0,:]}\n')
    print(f'All elements at position 1 from first axis (all elements from second row): \n{array_two[1,:]}\n')
    print(f'All elements at position 2 from first axis (all elements from third row): \n{array_two[2,:]}\n')
    print(f'All elements at position 0 from second axis (all elements from first  column): \n{array_two[:,0]}\n')
    print(f'All elements at position 1 from second axis (all elements from second column): \n{array_two[:,1]}\n')
    print(f'All elements at position 2 from second axis (all elements from third  column): \n{array_two[:,2]}\n')
    print(f'Elements at position 0 and 1 from first axis (rows) and position 0 from second axis (cols): \n{array_two[0:2,0]}\n')
    print(f'Elements at position 0 and 1 from first axis (rows) and position 0 and 1 from second axis (cols): \n{array_two[0:2,0:2]}\n')
    print(f'Elements at position 1 and 2 from first axis (rows) and position 1 and 2 from second axis (cols): \n{array_two[1:3,1:3]}\n')
    # slice with stride
    print(f'All elements at position 0 from first axis (all elements from first row) with stride 1: \n{array_two[0,::1]}\n')
    print(f'All elements at position 0 from first axis (all elements from first row) with stride 2: \n{array_two[0,::2]}\n')
    print(f'All elements at position 0 from second axis (all elements from first  column) with stride 1: \n{array_two[::1,0]}\n')
    print(f'All elements at position 0 from second axis (all elements from first  column) with stride 2: \n{array_two[::2,0]}\n')
    # three dimensional array
    print(f'First two-dimensional array:\n{array_three[0]}\n')
    print(f'Second two-dimensional array:\n{array_three[1]}\n')
    print(f'Elements at position 0,0,2 of 3-D array : \n{array_three[0,0,1]}\n')
    print(f'All elements at position 0 from first two-dimensional array (first row first array):\n{array_three[0][0,:]}\n')
    print(f'All elements at position 1 from second two-dimensional array (second row second array):\n{array_three[1][1,:]}\n')
