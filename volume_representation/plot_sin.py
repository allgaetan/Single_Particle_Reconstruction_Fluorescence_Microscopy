import matplotlib.pyplot as plt
import numpy as np

Te = 0.0001
X = np.arange(0,2*np.pi*5,0.0002)
plt.plot(X, 0.3*np.sin(X), color='blue', linewidth=10)

plt.show()