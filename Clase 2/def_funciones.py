# Definicion de funciones 
def prueba(a,b,c):
    import sympy as sp
    x = sp.Symbol('x')
    f = a*x**2 + b*x + c
    return f

def derivadaF(g):
    import sympy as sp
    salida1 = sp.diff(g,x,1)
    salida2 = sp.diff(g,x,2)
    return (salida1,salida2)

type(prueba(1,1,1))

p = derivadaF(prueba(1,1,1))

type(p[0])

p[0].subs(x,1)

p[1]
