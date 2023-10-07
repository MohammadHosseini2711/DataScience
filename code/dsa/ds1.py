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

# imagine a row of boxes
one_dim = np.array([1,2,3,4])
print(f'one dimension array:\n{one_dim}')

# imagine it as a n*m grid of boxes in a surface
two_dim=np.array([
    [1,2,3,4],[6,7,8,9]
])
print(f'two dimension array:\n{two_dim}')

# imagine it as a (n*m)*k grid of boxes in a Volume
three_dim=np.array([[
    [[1,2,3,4],[6,7,8,9]],[[10,11,12,13],[14,15,16,17]]
]])
print(f'two dimension array:\n{three_dim}')


