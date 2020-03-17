import numpy as np

import scipy.linalg as spla

help(spla.det)


import scipy.io as spio
help(spio.mminfo)


%cd "C:\Users\Administrador\Desktop\Clase2"

spio.mminfo("mahindas.mtx")

mahindas = spio.mmread("mahindas.mtx")
type(mahindas)
mahindas.shape

mahindas_Densa=mahindas.todense()
mahindas_Densa[1:10,1:10]
spla.inv(mahindas_Densa)
spla.det(mahindas_Densa)


def gen_ex(d0):
    x = np.random.randn(d0,d0)
    return x.T + x

mat1 =gen_ex(10**3)

i_mat1 = spla.inv(mat1)

np.allclose(np.dot(mat1, i_mat1), np.eye(10**3))

mat2 = gen_ex(100)

P,L,U = spla.lu(mat2)
P
L
U


mat2 == np.dot(L,U)
L @ U == mat2
np.allclose(mat2 , np.dot(L,U))


Q,R  = spla.qr(mat2)
Q
R

import scipy.optimize as spot
def f(X):
    x,y = X
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

f((0,0))

# minimo 1 
min1 = spot.minimize(f, (0,0), 
                     method = "SLSQP") 
type(min1)
min1["fun"]
min1["x"][0]
spla.det(min1["hess_inv"])

spla.eig(min1["hess_inv"])

help(spopt.minimize)



# minimo 2
spot.minimize(f, (3,-2)) 

# minimo 3
spot.minimize(f, (-3,-3)) 


# minimo 4
spot.minimize(f, (-3,3)) 

































