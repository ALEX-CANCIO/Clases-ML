import sympy as sp

F1 = 1 
F2 = 1
i = 3
Fibo1 = [F1,F2] 
while i <= 10:
    Fn = F1+F2
    F2 = F1
    F1 = Fn
    Fibo1.append(Fn)
    i = i+1


a1 = (1+sp.sqrt(5))/(2)
a2 = (1-sp.sqrt(5))/(2)
type(a1)
type(a2)
Fibo2 = []
for i in range(1,11):
    a = int((((a1**i - a2**i)/sp.sqrt(5))).evalf())
    Fibo2.append(a)
    

Fibo1 == Fibo2

for i in range(0,len(Fibo1)):
    print(Fibo1[i] == Fibo2[i])















