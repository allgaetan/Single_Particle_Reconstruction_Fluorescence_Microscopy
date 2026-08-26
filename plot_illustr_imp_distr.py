import matplotlib.pyplot as plt
import numpy as np
from manage_matplotlib.graph_setup import set_up_graph
from volume_representation.gaussian_mixture_representation.GMM_grid_evaluation import one_d_gaussian

nb_angles = 360
size = 20
nb_samples = 21
prob = np.array([1/nb_angles] * nb_angles)

#samples = np.random.choice(range(nb_angles), p=prob, size=nb_samples)
samples = np.linspace(0,360,nb_samples)
print(samples)
samples[0] +=2
samples[-1] -=2
for i in range(1, len(samples)-1):
    samples[i] = samples[i] + np.random.randint(-8,8)
print(samples)
samples = [ 2., 10., 29., 53., 65. , 86. ,113. ,128. ,145. ,157. ,184. ,202. ,222. ,227.,
 252. ,271. ,287. ,307. ,331. ,342. ,358.]
fig_size = 24


def plt_g(lab=False):

    set_up_graph(MEDIUM_SIZE=fig_size, SMALLER_SIZE=fig_size)
    if lab:
        plt.scatter(samples, [0]*nb_samples, marker='X', c='red', label=r"échantillons d'indices $\mathcal{I}^{l, \psi}$ tirés aléatoirement")
    else:
        plt.scatter(samples, [0] * nb_samples, marker='X', c='red')
    plt.xlim(0, 360)
    plt.scatter(samples, [0] * nb_samples, marker='X', c='red', s=100)
    plt.xlabel(r'angle $\psi$ (°)')
    plt.grid()
    plt.legend()


plt_g(lab=True)
plt.plot(range(nb_angles), prob, label=r"distribution d'importance $\mathcal{Q}^{l, \psi}$ initiale")
plt.ylabel('probabilités')
plt.ylim(-10**-4, 5*10**-3)
plt.legend()
plt.show()


values = [10**-2, 0.05, 10**-2, 0.09, 0.2, 0.3, 0.4, 0.5, 0.9, 0.62, 0.5, 0.3, 0.04, 0.01, 0.03, 0.01, 0.1, 0.3, 0.7,0.4,0.1]

def plt_h(lg=False):
    plt_g()
    plt.ylabel(r'vraisemblances')
    for i in range(nb_samples):
        if i == 0 and lg:
            plt.vlines(samples[i], 0, values[i], label=r"vraisemblances évaluées : $\forall i \in \mathcal{I}^{l,\psi} \: \: \tilde{\pi}^{l,\psi}_i = \sum_{j \in \mathcal{I}^{l,d}} \frac{p^l_{i,j}}{\mathcal{Q}^{l,d}_j}$")
        else:
            plt.vlines(samples[i], 0, values[i])
plt_h(True)
plt.legend()
plt.ylim(-0.05, 1.15)
plt.show()


K_rot = np.zeros((nb_samples, nb_angles))

gaussian_kernel = True
psis = np.linspace(0, 360, nb_angles)
for k in range(nb_samples):
    if gaussian_kernel:
        K_rot[k, :] = one_d_gaussian(psis, samples[k], 7)
    else:
        K_rot[k, :] = np.exp(np.cos(samples[k] - psis) * 5)


def update_imp_distr(imp_distr, phi, K, prop, M):
    # phi = phi ** (1 / temp)
    q_first_comp = phi @ K
    q_first_comp_norm = q_first_comp/np.sum(q_first_comp)
    prob = (1 - prop) * q_first_comp_norm + prop * np.ones(M) / M
    return q_first_comp, q_first_comp_norm, prob


values = np.array(values)
q_first_comp, q_first_comp_norm, prob = update_imp_distr(prob, values, K_rot, 0.5, nb_angles)
plt_h()
plt.plot(range(nb_angles), q_first_comp, c='green', label=r"vraisemblances $\breve{\pi}^{l,\psi}$ (non normalisées) interpolées par une méthode à noyau")
plt.ylim(-0.05,1.3)
plt.legend()




def plt_i(lg=False):
    set_up_graph(MEDIUM_SIZE=fig_size, SMALLER_SIZE=fig_size)
    plt.xlabel(r'angle $\psi$ (°)')
    plt.ylabel('vraisemblances')
    plt.grid()
    if lg:
        plt.plot(range(nb_angles), q_first_comp_norm, c='green', label=r"vraisemblances $\breve{\pi}^{l,\psi}$ (normalisées)")
    else:
        plt.plot(range(nb_angles), q_first_comp_norm, c='green')
    plt.legend()

angles_selected = np.random.choice(range(nb_angles), p=prob, size=nb_samples)
plt_i(True)
plt.plot(range(nb_angles), prob, c='blue', label=r"$\mathcal{Q}^{l,\psi} = \alpha \mathcal{U}^{\psi} + (1-\alpha) \breve{\pi}^{l,\psi} $")
plt.scatter(angles_selected, [0]*nb_samples, c='red', marker='X', label="tirages de l'itération suivante", s=100)
plt.xlim(0,360)
plt.legend()
plt.show()


"""
set_up_graph(MEDIUM_SIZE=fig_size, SMALLER_SIZE=fig_size)
plt.plot(range(nb_angles), prob, c='blue')
plt.grid()
plt.xlabel(r'angle $\psi$ (°)')
plt.ylabel('vraisemblances')
plt.legend()
plt.show()
"""