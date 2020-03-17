import sympy as sp
import sys 

#num_der = int(input("NUmero de derivadas "))

num_der = int(sys.argv[1])
x = sp.symbols('x')
f = sp.sin(x**2+sp.exp(x))

i = 1
while i <= num_der:
    print(sp.diff(f,x,i))
    i = i+1
    
# defino una lista vacia 
Cdegrees = []
# conjunto de puntos desde -20 a 40
a = -20
while (a <= 40):
    Cdegrees.append(a)
    a = a + 5
Fdegress = []
for C in Cdegrees:
    F = 9/5*C+32
    Fdegress.append(F)
    print(C,F)