import scipy as sp
import matplotlib.pyplot as plt

def f(xx):
    return xx**4 - 22*xx**2 + xx + 114

xx = np.arange(-5.0, 5.0, 0.1)

plt.figure()

plt.plot(xx, f(xx), 'k')
plt.show()

#primer minimo
minimize(f, (-2))

#segundo minimo
minimize(f, (2))