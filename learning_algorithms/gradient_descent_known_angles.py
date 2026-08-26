import numpy as np
from time import time
from common_image_processing_methods.rotation_translation import get_rotation_matrix
from manage_files.read_save_files import make_dir


def stopping_criteria(energies, eps, n=4, m=2):
    if len(energies) <= n:
        return False
    mean_current = np.mean(energies[-m:])
    mean_old = np.mean(energies[-n:-n + m])
    return mean_old - mean_current < eps


def gradient_descent_known_rot(volume_representation, true_rot_vecs, true_trans_vecs, views, params_learning_alg, known_trans, save_fold=''):#, save_fold):
    itr = 0
    recorded_energies = []
    nb_views = len(views)
    recorded_shifts = [[] for _ in range(nb_views)]
    if save_fold != '':
        sub_dir = f'{save_fold}/intermediar_results'
        make_dir(sub_dir)
    while itr < params_learning_alg.N_iter_max and (not stopping_criteria(recorded_energies, params_learning_alg.eps)):
        itr += 1
        total_energy = 0
        output_name = f'recons_epoch_{itr}'
        # if itr%10 == 0:
        # volume_representation.register_and_save(sub_dir, output_name, ground_truth_path=None)
        for v in range(nb_views):
            rot_mat = get_rotation_matrix(true_rot_vecs[v], params_learning_alg.convention)
            energy = volume_representation.one_gd_step(rot_mat, true_trans_vecs[v], views, params_learning_alg.lr, known_trans, v,
                                                      recorded_shifts) # (v==nb_views-1))
            total_energy += energy
        total_energy /= nb_views
        print(f"total energy ep {itr}", total_energy)
        recorded_energies.append(total_energy)
    return volume_representation, recorded_energies, recorded_shifts, itr
