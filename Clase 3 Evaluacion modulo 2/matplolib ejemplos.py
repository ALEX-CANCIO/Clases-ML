import numpy as np
Anomalias = np.genfromtxt("Anomalias-1880-2017.csv",
                          delimiter=",",
                          skip_header=5,
                          dtype=np.float64)
Anomalias.shape
import matplotlib.pyplot as plt
plt.plot(Anomalias[:,1])
plt.ylabel('Anomalias')
plt.show()


import matplotlib.pyplot as plt
Anomalias[:,0].min()
Anomalias[:,0].max()
Anomalias[:,1].min()
Anomalias[:,1].max()
plt.plot(Anomalias[:,0],Anomalias[:,1],
         'ro')
plt.axis([1877, 2020, -1, 1.1])
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# dominio
t = np.arange(0., 5., 0.2)
# rojo, azul y verde
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()


import matplotlib.pyplot as plt
import numpy as np
# Generamos data con ruido
x = np.linspace(0, 5, 30)
y = x ** 2 + np.exp(np.random.rand(30))
# Funciones para configurar el grafico
plt.figure()
plt.plot(x, y, 'r')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Funcion cuadratica + Ruido')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 5, 30)
y = x ** 2 + np.exp(np.random.rand(30))
plt.figure()
# ================================================
plt.subplot(1,2,1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('title X vs Y')
plt.plot(x, y, 'r--')
# ================================================
plt.subplot(1,2,2)
plt.xlabel('y')
plt.ylabel('x')
plt.title('title Y vs X')
plt.plot(y, x, 'g*-');
# ================================================
plt.show()




import numpy as np
import matplotlib.pyplot as plt
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)
t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)
plt.figure(1)
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()






























