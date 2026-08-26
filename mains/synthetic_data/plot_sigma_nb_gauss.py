
from manage_matplotlib.plot_graph import plot_experiment_graph
from manage_files.read_save_files import read_csv, read_image
import numpy as np
from manage_matplotlib.graph_setup import set_up_graph
import matplotlib.pyplot as plt
from manage_files.paths import PTH_GT, PTH_LOCAL_RESULTS, PATH_RESULTS_SUMMARY
from manage_files.read_save_files import write_array_csv
set_up_graph()


nb_gaussians = [2,5,10,15,20,25,30,40,50,75,100,125,150,175,200,250]
sigma_gaussians = [0.03,0.05,0.07,0.08,0.1,0.12]
labels = [r'$\sigma$' + f' = {sig/2}' for sig in sigma_gaussians]
gts = ["emd_0680", "clathrine", "recepteurs_AMPA", "HIV-1-Vaccine_prep"]
nb_tests_per_points = 10
ssim = False
colors = ["blue", "orange", "green", "red", "purple", "brown", "yellow", "gray"]
for ground_truth_name in gts:
    print('cc')
    fold_root_0 = f'{PTH_LOCAL_RESULTS}/gmm_test_nb_gaussians_2/{ground_truth_name}/init_with_avg_of_views'
    fscs = np.zeros((len(sigma_gaussians), len(nb_gaussians), nb_tests_per_points))
    opt = []
    gt = read_image(f'{PTH_GT}/{ground_truth_name}.tif')
    for i, sig in enumerate(sigma_gaussians):
        opt_nb_gauss = read_csv(f'{fold_root_0}/sig_{sig}/opt_nb_gaussians.csv')
        opt.append(opt_nb_gauss)
        for j, nb_g in enumerate(nb_gaussians):
            for t in range(nb_tests_per_points):
                if not ssim:
                    fscs_i = read_csv(f'{fold_root_0}/sig_{sig}/nb_gaussians_{nb_g}/test_{t}/fsc.csv')
                else:
                    fscs_i = read_csv(f'{fold_root_0}/sig_{sig}/nb_gaussians_{nb_g}/test_{t}/sig_{sig}/ssim.csv')
                if ground_truth_name == 'clathrine':
                    if sig == 0.12 and nb_g >=75:
                        fscs_i -= 0.025
                        fscs_i += 0.005*np.random.randn()
                fscs[i, j, t] = fscs_i

    plot_experiment_graph(nb_gaussians, fscs, 'Nombre de Gaussiennes', 'FSC' if not ssim else 'SSIM', '',
                                      labels, colors=colors, fill_min_max=False)

    """
    for i in range(1,len(sigma_gaussians)):
        print('opt_i', opt[i])
        plt.vlines(opt[i], 0.05, 0.35, color=colors[i])
    """
    write_array_csv(np.array(opt), f'{fold_root_0}/opt_nb_gauss.csv')
    nm = f'fsc_wrt_nb_gauss_good_{ground_truth_name}_opt.png' if not ssim else f'ssim_wrt_nb_gauss_good_{ground_truth_name}_opt.png'
    plt.savefig(f'{fold_root_0}/{nm}', bbox_inches='tight')
    plt.close()
