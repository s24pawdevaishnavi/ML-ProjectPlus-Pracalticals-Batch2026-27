# Practical No. 2
# Title: Basic implementation of python modules (numpy, pandas, sklearn)

list = [3, 1, 10, 9, 12, 6, 'jupyter']
print(type(list))
print(list)

import numpy as np
arr = np.array([10, 11, 2005])
print(type(arr))
print(arr)

import numpy as np
arr = np.array([8, 6, 3, 5, 7], dtype=int)
print(type(arr))
print(arr.dtype)

import numpy as np
arr = np.array([7, 3, 67, 8, 'home'])
print(type(arr))
print(arr.dtype)

import numpy as np
arr = np.array([2, 56, 8, 4, '39'], dtype=str)
print(type(arr))
print(arr.dtype)

import numpy as np
arr = np.array([[65757], [62656]])
print(arr.ndim)

import numpy as np
arr = np.array([[[86]]])
print(arr.ndim)

import numpy as np
arr = np.array([[9, 3, 6, 12], [10, 11, 20, 5]])
print(arr.ndim)

import numpy as np
arr = np.array([[9, 5, 1], [6, 2, 7], [7, 6, 1]])
print(arr.ndim)

v1 = np.array([8, 4, 3, 5, 2])
print(v1[2])
print(v1[3:6:1])

v1 = np.array([4, 2, 8, 6, 3, 1])
print(v1[5])
print(v1[2:6:3])
print(len(v1))

v1 = np.array([10, 11, 20, 28, 45, 61, 74, 12, 86])
print(v1[6])
print(v1[1:3:1])
print(v1.shape)

v1 = np.array([12, 89, 56, 74, 36, 89, 19, 28, 37, 85, 21, 64])
print(v1[1])
print(v1[1:6:1])
print(v1.reshape(2, 2, 3))

v1 = np.array([8, 4, 3, 1, 5, 94, 47, 71, 92, 63, 12, 36])
print(v1.reshape(2, 2, 3))
v2 = v1.flatten()
print(v2)

v1 = np.array([15, 25, 33, 14, 58, 90, 74, 11, 22, 13, 32, 76])
print(v1.reshape(2, 2, 3))
v2 = v1.flatten()
y = v2.tolist()
print(y)

v1 = np.array([10, 12, 93])
v2 = np.array([14, 52, 61])
x1 = v1 + v2
y2 = v1 - v2
print(x1)
print(y2)

v1 = np.array([11, 13, 20, 47, 51, 63])
v2 = np.array([94, 15, 67, 70, 12, 82])
x1 = v1 * v2
y2 = v1 / v2
print(x1)
print(y2)

v1 = np.zeros(100, dtype=int)
print(v1)

v1 = np.ones(101, dtype=int)
print(v1)

v1 = np.full(11, 20)
print(v1)

v1 = np.arange(20, 46)
print(v1)

v1 = np.identity(5, dtype=int)
print(v1)

a1 = np.random.rand(11)
print(a1)
a2 = np.random.rand(3, 2)
print(a2)
a3 = np.random.randint(10, 100, size=10)
print(a3)
a4 = np.random.randn(10)
print(a4)
a5 = np.random.ranf(5)
print(a5)

import seaborn as sns
array = np.random.randn(126)
v1 = np.random.randn(1)
v2 = np.random.randn(20)
x1 = sns.kdeplot(array)
print(x1)


"""
=================== OUTPUT (text output; plots open in a window when run normally) ===================
<class 'list'>
[3, 1, 10, 9, 12, 6, 'jupyter']
<class 'numpy.ndarray'>
[  10   11 2005]
<class 'numpy.ndarray'>
int64
<class 'numpy.ndarray'>
<U21
<class 'numpy.ndarray'>
<U2
2
3
2
2
3
[5 2]
1
[8 1]
6
74
[11 20]
(9,)
89
[89 56 74 36 89]
[[[12 89 56]
  [74 36 89]]

 [[19 28 37]
  [85 21 64]]]
[[[ 8  4  3]
  [ 1  5 94]]

 [[47 71 92]
  [63 12 36]]]
[ 8  4  3  1  5 94 47 71 92 63 12 36]
[[[15 25 33]
  [14 58 90]]

 [[74 11 22]
  [13 32 76]]]
[15, 25, 33, 14, 58, 90, 74, 11, 22, 13, 32, 76]
[ 24  64 154]
[ -4 -40  32]
[1034  195 1340 3290  612 5166]
[0.11702128 0.86666667 0.29850746 0.67142857 4.25       0.76829268]
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
[20 20 20 20 20 20 20 20 20 20 20]
[20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43
 44 45]
[[1 0 0 0 0]
 [0 1 0 0 0]
 [0 0 1 0 0]
 [0 0 0 1 0]
 [0 0 0 0 1]]
[0.88524587 0.72534742 0.81820949 0.37790924 0.477616   0.70486917
 0.27196098 0.23740807 0.01595637 0.63295238 0.01742742]
[[0.05736223 0.50288032]
 [0.59554442 0.36255178]
 [0.03485872 0.87561073]]
[35 33 81 32 29 20 34 69 61 10]
[ 0.74130346 -0.67200235 -0.11456089  0.51221088  1.67941958  0.80441589
 -1.41920389  0.80295316 -0.22899637 -0.35224899]
[0.14936848 0.72314044 0.6848379  0.64767276 0.13714923]
Axes(0.125,0.11;0.775x0.77)
"""
