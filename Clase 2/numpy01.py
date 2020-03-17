# importamos el modulo numpy
import numpy as np
import sympy  as sp
import scipy as scp
import matplotlib as mpl
import sys




help(np.sum)

dir(np)

help(np.array)

Cdegress = range(-20,41,5)
Fdegress = [9/5*C + 32 for C in Cdegress]

vector1 = np.array(Cdegress)
vector2 = np.array(Fdegress)

type(vector1)


NormaV1 = np.sqrt(np.sum([v**2 for v in vector1]))

def NormaP(u , p):
    return (np.sum([v**p for v in u]))**(1/p)

z1 = NormaP(vector2,3)


# Todas las normas de vector2 desde p=3
# hasta p=300
lista = []
for i in range(3,301):
    n = NormaP(vector2,i)
    if n == np.inf:
        print(i)
    lista.append(n)

def collatz1(n):
    if (n % 2 ==0):
        c = n/2
    else:
        c = 3*n+1
    return int(c)

collatz1(13)

n= 2**40
j=1
c = collatz1(n)
lista_c = [c]
while c != 1:
    c = collatz1(c)
    lista_c.append(c)
    j = j+1
























