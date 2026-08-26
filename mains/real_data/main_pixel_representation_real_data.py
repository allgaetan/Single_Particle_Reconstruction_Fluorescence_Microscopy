from learning_algorithms.gradient_descent_importance_sampling import gd_importance_sampling_3d
from learning_algorithms.gradient_descent_known_angles import gradient_descent_known_rot
from volume_representation.pixel_representation import Fourier_pixel_representation
from classes_with_parameters import ParametersMainAlg
from manage_files.read_save_files import read_image, read_images_in_folder, read_csv
from manage_files.paths import *
from common_image_processing_methods.rotation_translation import discretize_sphere_uniformly
import numpy as np
from manage_files.read_save_files import make_dir
from common_image_processing_methods.others import crop_center
from metrics_and_visualisation.plot_conical_fsc import plot_conical_fsc, fsc, plot_resolution_map
from metrics_and_visualisation.metrics_to_compare_2_images import plot_fsc_graph
from manage_matplotlib.graph_setup import set_up_graph
from manage_matplotlib.plot_graph import plot_graphs
import matplotlib.pyplot as plt


nb_dim = 3
size = 50
pth_inter = f'{PATH_PROJECT_FOLDER}/real_data'
pth_views = f'{pth_inter}/c1'

#pth_rot_vecs = f'{PATH_RESULTS_SUMMARY}/real_data/FSC/test_0/intermediar_results/estimated_rot_vecs_epoch_1.csv'
#pth_rot_vecs = f'{PATH_PROJECT_FOLDER}/results/real_data_centriole/intermediar_results/estimated_rot_vecs_epoch_49.csv'
#rot_vecs = read_csv(pth_rot_vecs)
psf = read_image(f'{pth_inter}/PSF_6_c1_resized_ratio_2.tif')
psf = crop_center(psf, (size, size, size))
views, file_names = read_images_in_folder(pth_views)
volume_representation = Fourier_pixel_representation(nb_dim, size, psf)
params_learning_alg = ParametersMainAlg(eps=-10, N_iter_max=51, lr=1)

uniform_sphere_discretization = discretize_sphere_uniformly(params_learning_alg.M_axes, params_learning_alg.M_rot)


for t in range(20):
    print('len', len(views))
    first_set_indices = np.random.permutation(range(len(views)))[:len(views)//2]
    second_set_indices = []
    for i in range(len(views)):
        if i not in first_set_indices:
            second_set_indices.append(i)

    views_splitted = [views[:len(views)//2], views[len(views)//2:]]

    for i in range(2):
        save_fold = f'{PATH_PROJECT_FOLDER}/results/real_data/FSC_many_sets/FSC_{t}/test_{i}'
        make_dir(save_fold)
        views_set = views_splitted[i]
        imp_distrs_rot = np.ones((len(views_set), params_learning_alg.M_rot)) / params_learning_alg.M_rot
        imp_distrs_axes = np.ones((len(views_set), params_learning_alg.M_axes)) / params_learning_alg.M_axes
        true_trans_vecs = np.zeros((len(views_set), 3))
        gd_importance_sampling_3d(volume_representation, uniform_sphere_discretization, true_trans_vecs, views_set, imp_distrs_axes,
                                      imp_distrs_rot, 1,  params_learning_alg.prop_min, params_learning_alg,
                                  False, save_fold, use_gpu=True)


pixel_size = 25
avg_fsc = []
avg_conical_fsc = []
avg_conical_fsc_main_axes = []
pth_o =  f'{PATH_PROJECT_FOLDER}/results/real_data/FSC_many_sets'
for t in range(19):
    print('t', t)
    pth_root = f'{pth_o}/FSC_{t}'
    pth1 = f'{pth_root}/test_0/intermediar_results/recons_epoch_50.tif'
    pth2 = f'{pth_root}/test_1/intermediar_results/recons_epoch_50.tif'
    im1 = read_image(pth1)
    im2 = read_image(pth2)
    radiuses_frequencies_1, fscs = fsc(im1, im2, plot_path=f'{pth_root}/fsc_curve.png', pixel_size=pixel_size)
    avg_fsc.append(fscs)
    radiuses_frequencies, thetas, phis, conical_fsc, conical_fsc_main_axes =(
         plot_conical_fsc(im1, im2, pth_root, nb_sectors=2000, pixel_size=pixel_size))
    avg_conical_fsc.append(conical_fsc)
    avg_conical_fsc_main_axes.append(conical_fsc_main_axes)

avg_fsc = np.array(avg_fsc)
avg_fsc = np.mean(avg_fsc, axis=0)
print('avg fsc shape', avg_fsc.shape)

plot_fsc_graph(radiuses_frequencies_1, avg_fsc, plot_path=f'{pth_o}/fsc_avg_curve.png', cutoff=0.143, pixel_size=pixel_size)

avg_conical_fsc_main_axes = np.array(avg_conical_fsc_main_axes)
avg_conical_fsc_main_axes = np.mean(avg_conical_fsc_main_axes, axis=0)

avg_conical_fsc = np.array(avg_conical_fsc)
avg_conical_fsc = np.mean(avg_conical_fsc, axis=0)

set_up_graph(MEDIUM_SIZE=70)
plot_resolution_map(radiuses_frequencies, thetas, phis, avg_conical_fsc, cutoff=0.143, vmin=0, vmax=radiuses_frequencies[-1] + radiuses_frequencies[1],
                            cmap_type="hot")

plt.savefig(f'{pth_o}/avg_cfc_map.png', bbox_inches='tight')
plt.close()


set_up_graph()
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plot_graphs(radiuses_frequencies, avg_conical_fsc.T, 'Frequence (1/nm)', 'Coefficient de corrélation', '', None,
            colors=['lightgray']*len(avg_conical_fsc.T), linewidth=7)

labels = ['Axe X', 'Axe Y',
              'Axe Z']
plot_graphs(radiuses_frequencies, avg_conical_fsc_main_axes.T, 'Frequence (1/nm)', 'Coefficient de corrélation',
            '', labels, linewidth=4)

plt.savefig(f'{pth_o}/avg_cfc_curve.png', bbox_inches='tight')
plt.close()





#gradient_descent_known_rot(volume_representation, rot_vecs, true_trans_vecs, views, params_learning_alg, False, save_fold)