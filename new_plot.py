import matplotlib.pyplot as plt
import numpy as np
from manage_matplotlib.plot_graph import plot_experiment_graph
from manage_matplotlib.graph_setup import set_up_graph

"""replot the graph ssim with nb gaussian to represent a view because the font size was too small"""

nb_gaussians = [5, 10, 12, 15, 17, 20, 30, 40, 50, 55, 58, 60, 62, 65, 70, 80, 90, 100]
SSIM_vals = [0.83, 0.862, 0.871, 0.876, 0.882, 0.902, 0.94, 0.96, 0.974, 0.975, 0.9748, 0.9745, 0.9744, 0.9748, 0.9742, 0.9738, 0.9733, 0.9729]
SSIM_vals = np.array(SSIM_vals) - 0.1
sz = 30
set_up_graph(MEDIUM_SIZE=sz, SMALLER_SIZE=sz)
plt.grid()
plt.plot(nb_gaussians, SSIM_vals)
plt.scatter(nb_gaussians, SSIM_vals, marker='X')
plt.rcParams['text.usetex'] = True
plt.vlines(66, 0.72, 0.88, label=r"Nombre de gaussiennes optimal théorique", color="red")
plt.rcParams['text.usetex'] = False
plt.legend()
plt.xlabel("Nombre de gaussiennes")
plt.ylabel("SSIM")
plt.show()

1/0



nb_gaussians = [5, 10, 20, 30, 40, 50, 60, 70, 80, 85, 87, 90, 92, 95, 97, 100, 110, 120, 130, 140, 150]
SSIM_vals = [0.551, 0.575, 0.63, 0.695, 0.73, 0.768, 0.78, 0.798, 0.8, 0.798, 0.797, 0.795, 0.793, 0.791, 0.788, 0.778,
             0.765, 0.759, 0.753, 0.748,0.747]
SSIM_vals = np.array(SSIM_vals)

sz = 30
set_up_graph(MEDIUM_SIZE=sz, SMALLER_SIZE=sz)
plt.grid()
plt.plot(nb_gaussians, SSIM_vals)
plt.scatter(nb_gaussians, SSIM_vals, marker='X')
plt.rcParams['text.usetex'] = True
plt.vlines(90, 0.55, 0.80, label=r"Nombre de gaussiennes optimal théorique", color="red")
plt.rcParams['text.usetex'] = False
plt.legend()
plt.xlabel("Nombre de gaussiennes")
plt.ylabel("SSIM")
plt.show()
1/0






