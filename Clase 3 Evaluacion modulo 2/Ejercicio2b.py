import scipy as sp
import matplotlib.pyplot as plt

polinomio = [1,0,-22,1,114]
x = sp.arange(-5,5,.05)
y = sp.polyval(polinomio,x)
raices = sp.roots(polinomio)
s = sp.polyval(polinomio,raices)
print ("Las raices son %2.2f, %2.2f, %2.2f, %2.2f. " % (raices[0], raices[1], raices[2],raices[3]))
plt.figure
plt.plot(x,y,'-', label = 'y(x)')
plt.plot(raices.real,s.real,'ro', label = 'Raices')
plt.xlabel('x')
plt.ylabel('y')
plt.title(u'Raices de un polinomio de x^3')
plt.legend()
plt.show()