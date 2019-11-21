import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc

xs = np.linspace(-50, 50, 1000)
t = 20

def cexp(x):
	return np.exp(-(x/t)**6)

def cabs(x):
	if np.abs(x) < t:
		return 1
	return 0

ys = cexp(xs)

zs = []
for x in xs:
	zs.append(cabs(x))

f = plt.figure()


rc('text', usetex=True)
# plot
plt.plot(xs, zs, 'r--')
plt.plot(xs, ys, 'b-')
# set labels (LaTeX can be used)
plt.xlabel(r'$I(r)-I(r_0)$', fontsize=13)
plt.ylabel(r'$c(I(r),I(r_0))$', fontsize=13)
plt.show()
f.savefig("cexp.png")