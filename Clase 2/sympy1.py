import sympy
from sympy import *

# Forma de cargar un modulo 
import sympy as sp
sp.init_printing(use_latex = False)

# alias para sympy es sp



# definicion de variables simbolicas
x , y, a, b = sp.symbols('x y a b')

type(x)

x + x 
x + 3*x 



f = a*x**3 + b*y**2 + a**b

type(f)

# definir variables simbolicas
sp.var('u , v')

type(u)


a = sp.Rational(1,2)
b = sp.Rational(3,4)
a**2 + b


ARacional = sp.Rational(2)**50/sp.Rational(10)**50
Afloat = 2**50/10**50

p = sp.pi**2
q = p.evalf()

import math as m
m.pi**2

z1 = sp.Symbol('z1')
z2 = sp.Symbol('z2')

(z1 + z2)**2

((z1 + z2)**2).expand()

((z1 + z2)**2).subs(z1,0)

((z1 + z2)**2).subs(z1,z2)

(((z1 + z2)**2).subs(z1,1)).subs(z2,2)

# infinito
sp.oo

# limite 
L1 = sp.limit(sp.sin(x)/x,x ,0)

L2 = sp.limit(1/x, x, sp.oo)

L3 = sp.limit(x**x ,x, 0)

sp.diff(x**2,x)

f = sp.sin(x**2+y**2)

# derivada parcial respecto de x 
sp.diff(f ,x)

sp.diff(f ,y)

gradF  = (sp.diff(f ,x),sp.diff(f ,y))


# expansion en series
g = sp.exp(sp.sin(x**2))

g.series(x,0,10)

(g**2).series(x,0,10)















