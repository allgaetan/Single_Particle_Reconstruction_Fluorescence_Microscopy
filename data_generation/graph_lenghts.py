from manage_matplotlib.graph_setup import set_up_graph
import matplotlib.pyplot as plt
import numpy as np

set_up_graph(MEDIUM_SIZE=30)
list_l_channel_1 = np.linspace(30, 350, 250)
list_l_channel_2 = np.array(list(np.linspace(30,100, 100)) + [100] * 150)
plt.scatter(range(250), list_l_channel_1, label="longueur canal 1", s=5, c='red')
plt.scatter(range(250), list_l_channel_2, label="longueur canal 2", s=5, c='green')
plt.xlabel("Etats de croissance")
plt.ylabel("Longueurs des canaux")
plt.legend()
plt.grid()
plt.show()