from manage_files.paths import PATH_PROJECT_FOLDER
from manage_files.read_save_files import read_4d
from metrics_and_visualisation.metrics_to_compare_2_images import compute_4d_metric, ssim, fsc
import numpy as np
from manage_files.read_save_files import write_array_csv, read_csv, save_4d_for_chimera
from common_image_processing_methods.registration import register_recons_4d, rotation_4d, translation4d
from manage_matplotlib.plot_graph import plot_experiment_graph
import matplotlib.pyplot as plt
from reconstruction_with_dl.pose_net import to_numpy
from common_image_processing_methods.others import resize
from common_image_processing_methods.rotation_translation import get_3d_rotation_matrix
from common_image_processing_methods.rotate_symmetry_axis import rotate_centriole_to_have_symmetry_axis_along_z_axis_4d
from manage_files.read_save_files import load_pickle
from data_generation.generate_data import heterogene_views_from_centriole
from reconstruction_with_dl.data_set_views import ViewsRandomlyOrientedSimData
import torch
from common_image_processing_methods.barycenter import center_barycenter_4d
from skimage.registration import phase_cross_correlation
from common_image_processing_methods.registration import registration_exhaustive_search


def register_im(gt, im, save_fold, id):
    registered_est_vol, R = rotate_centriole_to_have_symmetry_axis_along_z_axis_4d(im, axis_indice=2, slice_idx=id)

    rot_vec, reg_array = registration_exhaustive_search(gt, registered_est_vol[id, :, :, :],
                                                        '', '', 3,
                                                        save_res=False,
                                                        sample_per_axis=[360, 0, 0, 0, 0, 0])
    rot_mat = get_3d_rotation_matrix(np.degrees(rot_vec), convention='ZYX')

    registered_est_vol = rotation_4d(registered_est_vol, rot_mat)
    registered_est_vol = center_barycenter_4d(registered_est_vol)
    return registered_est_vol


def read_resize_gt(pth_gt, nb_channels):
    gt = []
    size = 50
    for c in range(nb_channels):
        gtc = read_4d(f'{pth_gt}/c{c + 1}/gt/4d_vol_channel.tiff')
        gtc_resized = []
        for v in range(gtc.shape[0]):
            gtc_resized.append(resize(gtc[v], (size, size, size)))
        gtc_resized = np.array(gtc_resized)
        gt.append(gtc_resized)
    return gt


def zbeul(pth_im, nb_channels, gt, nb_views, id=0, compute_only_corrcoeff=False):
    pth_vols = f'{pth_im}/vols'
    fsc_avg = 0
    ssim_avg = 0
    est_het = read_csv(f'{pth_im}/est_heterogeneities.csv')
    true_het = read_csv(f'{pth_im}/true_heterogeneities.csv')
    corrcoeff = np.corrcoef(np.array([est_het, true_het]))
    val_corrcoeff = np.abs(corrcoeff[0, 1])
    #reverse = (first_val >= last_val)
    if not compute_only_corrcoeff:
        for c in range(nb_channels):
            im_c = read_4d(f'{pth_vols}/4d_vol_channel_{c + 1}.tiff')
            reverse = np.sum(im_c[0]) > np.sum(im_c[-1])
            if reverse:
                im_c = im_c[::-1, :, :, :]
            # print('symmetry ax')
            true_vol_middle = gt[c][id, :, :, :]
            registered_est_vol = register_im(true_vol_middle, im_c, pth_vols, id)
            if nb_views <= 250:
                k = 249 / nb_views
                gt_c = gt[c][[int(i * k) for i in range(nb_views)]]
                gt_centered = center_barycenter_4d(gt_c)
            else:
                k = (nb_views-1) / 250
                gt_centered = center_barycenter_4d(gt[c])
                registered_est_vol = registered_est_vol[[int(i*k) for i in range(250)]]
            if nb_views < 250:
                save_4d_for_chimera(registered_est_vol, f'{pth_vols}/4d_vol_channel_{c + 1}.tiff')
            else:
                save_4d_for_chimera(registered_est_vol, f'{pth_vols}/4d_vol_channel_{c + 1}_reg.tiff')
                #save_4d_for_chimera(gt_centered, f'{pth_vols}/gt_centered_{c + 1}.tiff')
            fsc_val = compute_4d_metric(registered_est_vol, gt_centered, metric=fsc)
            fsc_avg += 1 / fsc_val
            ssim_val = compute_4d_metric(registered_est_vol, gt_centered, metric=ssim)
            ssim_avg += ssim_val

    fsc_avg /= nb_channels
    ssim_avg /= nb_channels
    write_array_csv(np.array([fsc_avg]), f'{pth_im}/fsc.csv')
    write_array_csv(np.array([ssim_avg]), f'{pth_im}/ssim.csv')
    write_array_csv(np.array([val_corrcoeff]), f'{pth_im}/corr_coeff_het.csv')
    return fsc_avg, ssim_avg, val_corrcoeff


if __name__ == '__main__':
    pth_results = f'{PATH_PROJECT_FOLDER}/results_deep_learning/centriole/results_hpc/test_snr/recons_results'
    pth_gt = f'{PATH_PROJECT_FOLDER}/results_deep_learning/heterogene_views/2_channels'
    nb_channels = 1
    gt = read_resize_gt(pth_gt, nb_channels)
    #snr_vals = [0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100]
    snr_vals = [10]
    nb_tests = 10

    corrs = np.zeros((len(snr_vals), nb_tests))
    fscs = np.zeros((len(snr_vals), nb_tests))
    ssims = np.zeros((len(snr_vals), nb_tests))

    for i,snr_val in enumerate(snr_vals):
        print('snr val', snr_val)
        for t in range(nb_tests):
            pth_im = f'{pth_results}/snr_{snr_val}/test_{t}/ep_2999'
            fsc_avg, ssim_avg, val_corrcoeff = zbeul(pth_im, nb_channels, gt)
            fscs[i, t] = fsc_avg
            ssims[i, t] = ssim_avg
            corrs[i, t] = val_corrcoeff

    plot_experiment_graph(snr_vals, [fscs], "Rapport signal à bruit", "FSC", "", [""])
    plt.xscale('log')
    plt.xticks(snr_vals)
    plt.savefig(f'{pth_results}/fscs.png')
    plt.close()

    plot_experiment_graph(snr_vals, [ssims], "Rapport signal à bruit", "SSIM", "", [""])
    plt.xscale('log')
    plt.xticks(snr_vals)
    plt.savefig(f'{pth_results}/ssims.png')
    plt.close()

    plot_experiment_graph(snr_vals, [corrs], "Rapport signal à bruit", "Corrélation hétérogénéité réelle/estimée", "", [""])
    plt.xscale('log')
    plt.xticks(snr_vals)
    plt.savefig(f'{pth_results}/corrs.png')
    plt.close()

    #write_array_csv(ssims, f'{pth_results}/ssims.csv')
    #write_array_csv(fscs, f'{pth_results}/fscs.csv')
    write_array_csv(corrs, f'{pth_results}/corrs.csv')

"""
def regenerate_views_from_saved_params(fold):
    params_data_gen = load_pickle(f'{fold}/params_data_gen')
    params_learn_setup = load_pickle(f'{fold}/params_learn_setup')
    true_rot_vecs = read_csv(f'{fold}/ep_2999/true_rots.csv')
    print('cc', true_rot_vecs[0])
    true_rot_vecs = np.array([np.degrees(true_rot_vecs[i]) for i in range(len(true_rot_vecs))])
    x = 100
    list_l_channel_1 = np.linspace(30, 350, params_data_gen["nb_views"])
    list_l_channel_2 = np.array(list(np.linspace(30, 100, x)) + [100] * (params_data_gen["nb_views"] - x))
    cs = [103.25, 50]
    nb_channels = params_learn_setup["nb_channels"]
    params_data_gen["nb_views"] = 1
    views, het_vals, rot_mats, rot_vecs, transvecs, gt_4d = heterogene_views_from_centriole(
        params_data_gen["pth_views"], params_data_gen, nb_channels,
        [list_l_channel_1, list_l_channel_2], cs=cs, rot_vecs=true_rot_vecs)
    file_names = ["" for _ in range(params_data_gen["nb_views"])]

    data_set = ViewsRandomlyOrientedSimData(views, rot_mats, true_rot_vecs, transvecs, het_vals[0],
                                            params_data_gen["size"], params_data_gen["nb_dim"], file_names)
    return data_set


def register_recons(fold, dataset):
    device = 0
    end_to_end_net = load_pickle(f'{fold}/saved_model')
    end_to_end_net.cuda(device)
    view_0 = dataset[0][1].unsqueeze(0).cuda(device)
    true_rot_mat_0 = dataset[0][3]
    est_rot_mat_0, est_trans_0, _ = end_to_end_net.pose_net.forward(view_0, test=True)
    print('true rot mat0', true_rot_mat_0)
    print('est rot mat 0', est_rot_mat_0)
    register_rot_mat = to_numpy(true_rot_mat_0.T) #@ to_numpy(est_rot_mat_0.squeeze())
    fold2 = f'{fold}/ep_2999/vols'
    for c in range(2):
        im_c = read_4d(f'{fold2}/4d_vol_channel_{c + 1}.tiff')
        print('im_c shape', im_c.shape)
        registerd_im_c = rotation_4d(im_c, register_rot_mat)
        save_4d_for_chimera(registerd_im_c, f'{fold2}/4d_vol_reg_{c+1}.tiff')



    #est_rot_mat_0 = est_rot_mat_0.cuda(device)
    #est_trans_0 = est_trans_0.cuda(device)


    for v, d in enumerate(dataset):
        _, view, _, rot_mat, rot_vec, _, dilatation_val, file_name = d
        view = view.cuda(device)
        rot_mat = rot_mat.cuda(device)
        _, est_vol, est_trans, _, est_rot_mat, est_heterogeneity, _, _ = end_to_end_net.forward(view.unsqueeze(0), torch.zeros(3),
                                                                               torch.eye(3), test=True, known_rot=True,
                                                                               known_trans=True)
        print('est rot mat', est_rot_mat)
        print('true rot mat', rot_mat)
        1/0



pth = '/home/eloy/Documents/stage_reconstruction_spfluo/results_deep_learning/centriole/results_hpc/test_snr/recons_results/snr_100/test_1/'
dataset = regenerate_views_from_saved_params(pth)
register_recons(pth, dataset)
1/0
"""