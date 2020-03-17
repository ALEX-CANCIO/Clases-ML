from numpy import *
import numpy as np

a = arange(10).reshape(2,5)
a.shape
a.ndim
a.size
a.dtype
a.itemsize


np.arange(5)
np.zeros((2, 3))
np.ones((3, 2), dtype=int)
np.empty((2, 2))
np.linspace(-np.pi, np.pi, 5)
np.array([-3.141592,-1.570796,1.570796,3.141592])
np.random.rand(4)
np.random.randn(4)
np.random.seed(1234)

a = arange(20, 60, 10)
a
a + 1
a * 2
a
a /= 2
a


a = np.arange(5)
a >= 3
a % 2 == 0
a = np.reshape(np.arange(16), (4,4))
large_values = (a > 10)
even_values = (a%2 == 0)

a = (np.random.rand(1,10))
result = np.zeros(10)
result = (a.reshape(1,10))
print ("a: \n" + str(a))
for i in range(10):
    if a[0,i]>0.5:
        result[0,i] = 1
    else:
        result[0,i] = 0
print ("result.: \n" +str(result))


a = (np.random.rand(1,10))
print ("a: \n" + str(a))
result = np.where(a>0.5,1,0)
print ("result: \n ", result)

