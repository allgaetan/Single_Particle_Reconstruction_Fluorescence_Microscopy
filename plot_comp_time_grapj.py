import numpy as np
from manage_matplotlib.graph_setup import set_up_graph
import matplotlib.pyplot as plt

N = 20
N_min = 50
N_d = 64
N_psi = 8
common_factor = N * N_min * N_d * N_psi
ratio = 5
print('f', common_factor)


def get_time_gauss(s, K=2.17*10**-7,xhi=0.73,N_g=100, sigma_p=0.03, sigma_z_p=0.1, sigma_xy_p = 0.02):
    K/=ratio
    sigma = sigma_p * s
    sigma_z = sigma_z_p * s
    sigma_xy = sigma_xy_p * s
    return 2*K*N_g* (27*(sigma + sigma_xy)**2*(sigma + sigma_z))**xhi


def get_time_corr_phase(s, L = 5.86*10**-9):
    L/=ratio
    return L * s**3 * np.log(s**3)


def get_time_rotation(s, K_1=10**-7):
    K_1/=ratio
    return  K_1 * s**3


list_s = np.arange(5,100,1)
times_gauss = common_factor * get_time_gauss(list_s) /3600
times_cor_phases = common_factor * get_time_corr_phase(list_s)/3600
times_rotations = common_factor * get_time_rotation(list_s)/3600

times_gauss_corr_phase = times_gauss + times_cor_phases
time_rotation_corr_phase = times_rotations + times_cor_phases

set_up_graph(MEDIUM_SIZE=30)
plt.plot(list_s, times_cor_phases, color='blue', label='temps corrélations de phase')
plt.plot(list_s, times_rotations, color='green', label='temps rotations')
plt.plot(list_s, time_rotation_corr_phase, color='red', label='temps total SHiReVol')
plt.grid()
plt.legend()
plt.xlabel("Taille d'un côté de l'image (pixel)")
plt.ylabel("Temps de calcul (heures)")
plt.show()

set_up_graph(MEDIUM_SIZE=30)
plt.plot(list_s, times_gauss, color='pink', label="temps calcul des gaussiennes")
plt.plot(list_s, times_cor_phases, color='blue', label='temps corrélations de phase')
plt.plot(list_s, time_rotation_corr_phase, color='red', label='temps total SHiReVol')
plt.plot(list_s, times_gauss_corr_phase, color='orange', label='temps total SHiReGMM')
plt.xlabel("Taille d'un côté de l'image (pixel)")
plt.ylabel("Temps de calcul (heures)")
plt.legend()
plt.grid()
plt.show()
