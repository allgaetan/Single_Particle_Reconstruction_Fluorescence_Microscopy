from metrics_and_visualisation.fourier_shell_correlation import (plot_resolution_map,
                                                                 conical_fourier_shell_correlation_split_computation, conical_fourrier_shell_correlation)
import numpy as np
from common_image_processing_methods.rotation_translation import (discretize_sphere_uniformly,
                                                                  conversion_2_first_eulers_angles_cartesian)
from manage_files.paths import PATH_VIEWS, PTH_LOCAL_RESULTS
from manage_files.read_save_files import read_image
from manage_matplotlib.graph_setup import set_up_graph
import matplotlib.pyplot as plt
from metrics_and_visualisation.metrics_to_compare_2_images import fsc
from manage_files.read_save_files import write_array_csv
from common_image_processing_methods.others import normalize
from manage_matplotlib.plot_graph import plot_graphs
from manage_files.read_save_files import make_dir, save_figure, save
from common_image_processing_methods.rotation_translation import rotation, get_3d_rotation_matrix


def plot_conical_fsc(im1, im2, save_fold, plot_all=True, pixel_size=None, nb_sectors=2000):
    make_dir(save_fold)
    nb_radius = 30
    sig_radus = 1
    coeff_kernel_axes = 2
    points_to_add = [(0, 0), (0, 180), (360, 0), (360, 180), (0, 90), (360, 90), (180, 180), (180, 0), (180, 90)]
    thetas, phis, _ = discretize_sphere_uniformly(nb_sectors - len(points_to_add))
    thetas = list(thetas)
    phis = list(phis)
    for i in range(len(points_to_add)):
        thetas.append(points_to_add[i][0])
        phis.append(points_to_add[i][1])
    if plot_all:
        x, y, z = conversion_2_first_eulers_angles_cartesian(np.array(thetas), np.array(phis))
        axes = np.array([x, y, z])
        labels = None
        print('before')
        conical_fsc, radiuses_frequencies = conical_fourier_shell_correlation_split_computation(im1, im2, axes,
                                                                                                nb_radius,
                                                                                                coeff_kernel_axes,
                                                                                                sig_radus, set_size=10)
        print('after')
        if pixel_size is not None:
            radiuses_frequencies /= pixel_size
        set_up_graph(MEDIUM_SIZE=70)

        plot_resolution_map(radiuses_frequencies, thetas, phis, conical_fsc, cutoff=0.143, vmin=0, vmax=radiuses_frequencies[-1] + radiuses_frequencies[1],
                            cmap_type="hot")
        plt.savefig(f'{save_fold}/cfc_map.png', bbox_inches='tight')
        plt.close()
        set_up_graph()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plot_graphs(radiuses_frequencies, conical_fsc.T, 'Frequence (1/pixel)', 'Coefficient de corrélation', '', labels,
                    colors=['lightgray']*len(conical_fsc.T), linewidth=7)
        write_array_csv(conical_fsc, f'{save_fold}/conical_fsc.csv')


    main_axes = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])
    labels = ['Axe X', 'Axe Y',
              'Axe Z']
    conical_fsc_main_axes, radiuses_frequencies = conical_fourrier_shell_correlation(im1, im2, main_axes, nb_radius, coeff_kernel_axes, sig_radus)
    if pixel_size is not None:
        radiuses_frequencies /= pixel_size
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plot_graphs(radiuses_frequencies, conical_fsc_main_axes.T, 'Fréquence (1/pixel)', 'Coefficient de corrélation', '', labels,
                linewidth=4)
    plt.grid()
    save_figure(save_fold, 'cfcs_graphs.png')

    # plt.savefig(f'{save_path}/cfc_graphs.png')
    plt.close()

    if plot_all:
        return radiuses_frequencies, thetas, phis, conical_fsc, conical_fsc_main_axes


if __name__ == '__main__':
    from manage_files.paths import PATH_RESULTS_SUMMARY, PTH_GT, PATH_RESULTS_HPC, PATH_VIEWS
    from manage_files.read_save_files import read_csv

    gts = ["recepteurs_AMPA", "HIV-1-Vaccine_prep","emd_0680"]
    pth_root = f'{PATH_RESULTS_SUMMARY}/cryo_ransac_reconstruction'
    for t in range(1):
        for part_name in gts:
            pth = f'{pth_root}/{part_name}/test_{t}/recons_registered_gradient.tif'
            pth_gt = f'{PTH_GT}/{part_name}.tif'
            recons = normalize(read_image(pth))
            gt = normalize(read_image(pth_gt))
            plot_conical_fsc(gt, recons, f'{pth_root}/{part_name}')


    pth_root = f'{PATH_RESULTS_SUMMARY}/fortun_reconstructions'

    for part_name in gts:
        pth = f'{pth_root}/{part_name}/recons_registered_gradient.tif'
        pth_gt = f'{PTH_GT}/{part_name}.tif'
        recons = normalize(read_image(pth))
        gt = normalize(read_image(pth_gt))
        plot_conical_fsc(gt, recons, f'{pth_root}/{part_name}')
    1/0


    pth_root = f'{PATH_VIEWS}/HIV-1-Vaccine_prep/single_view/sig_z_5'
    pth_view = f'{pth_root}/view_0.tif'
    pth_gt = f'{pth_root}/gt.tif'
    view = read_image(pth_view)
    gt = read_image(pth_gt)
    pth = f'{PATH_RESULTS_SUMMARY}/cfsc_views/HIV-1-Vaccine_prep_2'
    plot_conical_fsc(gt, view, pth)
    1/0

    pth = '/home/eloy/Documents/stage_reconstruction_spfluo/results/real_data/FSC'

    recons1 = read_image(f'{pth}/test_0/intermediar_results/recons_epoch_49.tif')
    recons2 = read_image(f'{pth}/test_1/intermediar_results/recons_epoch_49.tif')
    fsc(recons1, recons2, plot_path=f'{pth}/fsc_curve.png')
    plot_conical_fsc(recons1, recons2, pth)
    1/0

    gts = ["HIV-1-Vaccine_prep", "emd_0680", "recepteurs_AMPA"]

    """
    pth_root = f'{PATH_RESULTS_SUMMARY}/fortun_reconstructions'
    for part_name in gts:
        pth = f'{pth_root}/{part_name}/recons_registered_gradient.tif'
        pth_gt = f'{PTH_GT}/{part_name}.tif'
        recons = normalize(read_image(pth))
        gt = normalize(read_image(pth_gt))
        plot_conical_fsc(gt, recons, f'{pth_root}/{part_name}')

    """
    pth_root = f'{PATH_RESULTS_SUMMARY}/voxels_reconstruction/poses_connues'
    """
    for t in range(10,30):
        pth_fold = f'{pth_root}/HIV-1-Vaccine_prep/test_{t}'
        conical_fsc = read_csv(f'{pth_fold}/conical_fsc.csv')
        nb_sectors = 2000
        points_to_add = [(0, 0), (0, 180), (360, 0), (360, 180), (0, 90), (360, 90), (180, 180), (180, 0), (180, 90)]
        thetas, phis, _ = discretize_sphere_uniformly(nb_sectors - len(points_to_add))
        thetas = list(thetas)
        phis = list(phis)
        for i in range(len(points_to_add)):
            thetas.append(points_to_add[i][0])
            phis.append(points_to_add[i][1])
        set_up_graph(MEDIUM_SIZE=70)
        radiuses_frequencies = np.linspace(0, 0.5, 30)
        plot_resolution_map(radiuses_frequencies, thetas, phis, conical_fsc, cutoff=0.143, vmin=0, vmax=0.5,
                            cmap_type="hot")
        plt.savefig(f'{pth_fold}/cfc_map_2.png', bbox_inches='tight')
        plt.close()
    1/0
    """
    for i in range(len(gts)):
        all_conical_fsc = []
        for t in range(5,10):
            pth_root_2 = f'{pth_root}/{gts[i]}/test_{t}'
            pth_gt = f'{pth_root_2}/ground_truth.tif'
            pth_im = f'{pth_root_2}/recons.tif'
            gt = normalize(read_image(pth_gt))
            im = normalize(read_image(pth_im))
            radiuses_frequencies, thetas, phis, conical_fsc = plot_conical_fsc(gt, im, pth_root_2)

            all_conical_fsc.append(conical_fsc)
        all_conical_fsc = np.array(all_conical_fsc)
        avg_conical_fsc = np.mean(all_conical_fsc, axis=0)
        set_up_graph()
        plot_resolution_map(radiuses_frequencies, thetas, phis, avg_conical_fsc, cutoff=0.143, vmin=0, vmax=0.5,
                            cmap_type="hot")
        plt.savefig(f'{pth_root}/{gts[i]}/cfc_map.png', bbox_inches='tight')
        plt.close()



    """
    part_name = "emd_0680"
    PATH_RESULTS_HPC = "/home/eloy/Documents/archives/stage_reconstruction_spfluo/results_hpc"
    pth_folder_res = f"{PATH_RESULTS_HPC}/test_gt/{part_name}/test_1"
    sig_z = 10
    pth_view = f'{PATH_VIEWS}/{part_name}/single_view/sig_z_{sig_z}/view_10.tif'
    t = 3
    pth = ('/home/eloy/Documents/stage_reconstruction_spfluo/results_summary/voxels_reconstruction/'
                 f'poses_inconnues/{part_name}/test_3')
    pth_voxel = f'{pth}/recons_registered.tif'
    pth_voxel_registered = f'{pth}/recons_registered_gradient.tif'
    # pth_voxel = f'/home/eloy/Documents/archives/stage_reconstruction_spfluo/results/recepteurs_AMPA/known_angles_sig_z_5/test_{t}/recons.tif' #f'{pth_folder_res}/recons_registered.tif'
    pth_gmm = f'{PTH_LOCAL_RESULTS}/gmm_test/recepteurs_AMPA.tif/test_unknown_angles/recons.tif'
    pth_gt = f'/home/eloy/Documents/archives/stage_reconstruction_spfluo/results/{part_name}/known_angles_sig_z_5/test_{t}/ground_truth.tif'
    pth_gmm_known_poses = ('/home/eloy/Documents/archives/stage_reconstruction_spfluo/results/gmm_test_nb_gaussians_2/'
                           'recepteurs_AMPA/init_with_avg_of_views/sig_0.05/nb_gaussians_200/step_0.tif')

    im_voxel = normalize(read_image(pth_voxel))
    im_registered = normalize(read_image(pth_voxel_registered))

    slight_rotation = get_3d_rotation_matrix([1,1,1])
    im_voxel_slightlty_rotated, _ = rotation(im_voxel, slight_rotation)
    save('slightly_rotated.tif', im_voxel_slightlty_rotated)
    save('original.tif', im_voxel)

    im_gmm = normalize(read_image(pth_gmm))
    im_gt = normalize(read_image(pth_gt))
    im_gmm_known_poses = normalize(read_image(pth_gmm_known_poses))
    view = normalize(read_image(pth_view))

    fsc_voxel = fsc(im_voxel, im_gt, cutoff=cutoff, plot_path='./fsc_curve.png')
    fsc_registered = fsc(im_registered, im_gt, cutoff=cutoff)
    fsc_slightly_rotated = fsc(im_voxel_slightlty_rotated, im_gt)
    fsc_gmm = fsc(im_gmm, im_gt, cutoff=cutoff)
    fsc_gmm_known_poses = fsc(im_gmm_known_poses, im_gt, cutoff=cutoff)
    print('fsc voxel', fsc_voxel)
    print('fsc registered', fsc_registered)
    print('fsc gmm', fsc_gmm)
    print('fsc gmm known poses', fsc_gmm_known_poses)
    print('fsc slightly rotated', fsc_slightly_rotated)
    1/0

    # 1/0
    # plot_conical_fsc(im_gmm, im_gt, "cfc_gmm.png")
    # plot_conical_fsc(im_gmm_known_poses, im_gt, "cfc_gmm_known_poses.png")
    #plot_conical_fsc(im_voxel_slightlty_rotated, im_gt, "./known_poses_slightly_rotated")
    plot_conical_fsc(view, im_gt, "./view")
    """



