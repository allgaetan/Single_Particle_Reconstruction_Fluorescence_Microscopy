import matplotlib.colors as mcolors
import numpy as np

def gen_colors(nb_colors):
    colors = list(mcolors.BASE_COLORS.keys())[:-1]
    while len(colors) < nb_colors:
        c = np.random.random(3)
        colors.append(tuple(c))
    return colors