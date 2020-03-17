import sympy as sp


num_der = int(input("NUmero de derivadas "))
x = sp.symbols('x')
f = sp.sin(x**2+sp.exp(x))

i = 1
while i <= num_der:
    print(sp.diff(f,x,i))
    i = i+1
    
    
    
