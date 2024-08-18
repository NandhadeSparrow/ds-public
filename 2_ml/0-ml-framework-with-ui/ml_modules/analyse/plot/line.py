import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0,10,1000) # x is a 1000 evenly spaced numbers from 0 to 10

plt.plot(x, np.sin(x), 'k:', label='sin(x)')
plt.plot(x, np.cos(x), 'r--', label='cos(x)')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('function sin(x)')

plt.savefig("plot.png")
plt.show()
