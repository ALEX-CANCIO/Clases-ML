from sympy import *
import sympy as sp
sp.init_printing(use_latex = False)
Rm, Rf, r, Rs= sp.symbols('Rm Rf r Rs')
print('Ingrese parametros numero entre 0 y 1: ')
a, b, c = True, True, True
while(a):    
    Rm = input("Rendimiento medio de la cartera: ")        
    Rm = float(Rm)
    if(Rm < 0 or Rm >1):
        print('Introduzca un numero entre 0 y 1')
    else:
        a = False
  
while(b):
    Rf = input("Tasa media libre de riesgo: ")
    Rf = float(Rf)
    if(Rf < 0 or Rf >1):
        print('Introduzca un numero entre 0 y 1')
    else:
        b = False
   
while(c):   
    r = input("Riesgo de la cartera: ")
    r = float(r)    
    if(r <= 0 or r >1):        
        print('Introduzca un numero mayor a 0 y menor igual a 1')
    else:
        c = False


Rs = (Rm - Rf)/r 
print("El valor de Ratio Sharpe es %.2f" %(Rs))