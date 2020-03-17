from sympy import *
import sympy as sp

import matplotlib.pyplot as plt
from scipy.optimize import minimize


#Sympy
sp.init_printing(use_latex = False)
x = sp.Symbol('x')
fx = x**4 - 22*x**2 + x + 114
solve(fx,x)