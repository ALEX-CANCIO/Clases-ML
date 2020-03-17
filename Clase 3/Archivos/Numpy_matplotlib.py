import numpy as np

dir(np)

help(np.random.rand)

a = np.arange(20,60,10)
a +1 

a**2

a = np.reshape(np.arange(16),(4,4))

large_values = a>10

even_values = (a %2 ==0)

b = a%2==0
a[b]

dir(a)
a[:,2].min()
a[:,2].argmin()


lista = [1,2,"ABC"]
dir(lista)



a
c = a**2

type(a)
type(c)

a*c

np.mat(a)*np.mat(c)


0**0
0/0

vec1 = np.array([-1,0,1])
vec2 = vec1/0

np.isnan(vec2)

999**999

vec3 = np.array([999])
type(vec3)
vec3**vec3
vec3 = np.array([999],dtype = np.float64)
vec3**vec3

vec3 = np.array([999],dtype = np.float128)

vec4 = np.ones(5)
vec4[4] = np.nan

vec4.sum()

np.nansum(vec4)

help(np.apply_along_axis)

np.__config__.show()
np.__version__

%cd "C:\Users\Administrador\Desktop\Clase2"

A = np.loadtxt("matrix_a.dat")
A
type(A)
A.min()
A[:,1]
A[0,:]


%ls


data = np.genfromtxt("stockholm_td_adj.dat.txt")
data

Anomalias = np.genfromtxt("Anomalias-1880-2017.csv",
                          skip_header = 5,
                          delimiter = ",",
                          dtype= np.float64)
Anomalias
type(Anomalias)
Anomalias.shape[0]
Anomalias.shape[1]


Senamhi = np.genfromtxt("qc00000547.txt",
                        delimiter = " ",
                        dtype= np.float64)

Senamhi.shape

Anomalias.shape
import matplotlib.pyplot as plt
plt.plot(Anomalias[:,1])
plt.ylabel("Anomalias")
plt.show()


Anomalias[:,0].min()
Anomalias[:,0].max()
Anomalias[:,1].min()
Anomalias[:,1].max()
plt.plot(Anomalias[:,0], Anomalias[:,1],
         'ro')
plt.axis([1877,2020,-1,1.1])





x = np.linspace(0,5,30)
len(x)
y = x**2 + np.exp(np.random.rand(30))
z = np.sqrt(x)+ np.random.rand(30)
plt.figure()
plt.plot(x,y,'r', x, z, 'b^')
plt.ylabel('y')
plt.xlabel('x')
plt.title('Funcion cuadratica + Ruido')
plt.show()


Nvidia = np.genfromtxt("NVDA.csv",
                       delimiter = ",",
                       skip_header = 1)

Nvidia.shape
Nvidia[0,]

plt.figure()
# =============================================================================
# Primer grafico
# =============================================================================
plt.subplot(2,1,1)
plt.plot(Nvidia[:,1],'r--')
plt.xlabel("Meses")
plt.ylabel("Apertura")
plt.title("Empresa NVIDA [Mensual]")
# =============================================================================
# Segundo grafico 
# =============================================================================
plt.subplot(2,1,2)
plt.plot(Nvidia[:,2],'g*-')
plt.xlabel("Meses")
plt.ylabel("Alto")
#plt.title("Empresa NVIDA [Mensual]")
########################################
plt.savefig("Nvidia.png")
plt.show()













