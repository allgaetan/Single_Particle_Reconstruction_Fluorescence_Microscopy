import numpy as np
from common_image_processing_methods.others import resize
import torch
from pytorch3d.transforms import axis_angle_to_matrix
from common_image_processing_methods.rotation_translation import (
    discretize_sphere_uniformly, 
    get_3d_rotation_matrix, 
    get_rot_vec_from_rot_mat,
    rotation)
from stage_gaetan.hps_visualization import plot_search_round
from manage_files.paths import PATH_PROJECT_FOLDER
import skimage.io as io
from reconstruction_with_dl.pose_net import to_numpy
import torch.nn.functional as F
from manage_files.read_save_files import save

def rescale_volume(x, size=None, scale_factor=None):
    """
    Rescale un volume 3D avec soit un facteur d'échelle soit une taille cible
    - x: volume 3D à rescaler
    - size: taille cible pour le rescaling (tuple de 3 entiers)
    - scale_factor: facteur d'échelle pour le rescaling (float)
    """
    if scale_factor is not None:
        return F.interpolate(x, scale_factor=scale_factor, mode='trilinear', align_corners=False, recompute_scale_factor=True)
    if size is not None:
        return F.interpolate(x, size=size, mode='trilinear', align_corners=False)
    return x

def hierarchical_pose_search(end_to_end_net, 
                             index,
                             view, 
                             trans, 
                             het, 
                             params_learning_setup, 
                             params_data_gen, 
                             rot_mat_axis=None,
                             plot=False,
                             verbose=False):
    """
    Algorithme de recherche hiérarchique de pose (HPS) pour estimer la pose d'une vue donnée
    - end_to_end_net: architecture end to end de Fluofire
    - index: indice de la vue dans la table de correspondance
    - view: volume 3D de la vue à traiter
    - trans: vecteur de translation associé
    - het: paramètre de conformation associé
    - params_learning_setup: paramètres pour l'apprentissage
    - params_data_gen: paramètres pour la génération de données
    """ 
    # hyperparamètres et configuration
    bs = params_learning_setup["bs"]
    assert bs == view.shape[0]
    print(f"Batch size: {bs}")

    convention = params_data_gen["convention"]
    device=params_learning_setup["device"]
    nb_kept = params_learning_setup["nb_kept_candidates"]
    nb_view_dir = params_learning_setup["nb_view_dir"]
    nb_in_plane = params_learning_setup["nb_in_plane"]
    nb_children = params_learning_setup["nb_children"]
    max_iter = params_learning_setup["hps_max_iter"]
    sigma_start = params_learning_setup["sigma_start"]
    sigma_shrink = params_learning_setup["sigma_shrink"]
    save_fold = params_learning_setup["save_fold"] + "/pose_search"
    scale_factors = params_learning_setup["scale_factors"]
    mean_errors = []

    # phase d'initialisation
    if scale_factors:
        init_scale = scale_factors[0]
        init_view = rescale_volume(view, scale_factor=init_scale)
        print(f"View downsampled to {init_view.shape}")
        save(f'{params_learning_setup["save_fold"]}/downsampled_init.tiff', to_numpy(init_view))
    else:
        init_scale = None
        init_view = view

    base_grid = build_base_rotation_grid(nb_view_dir, nb_in_plane, convention, device)
    base_errors = evaluate_candidates(base_grid, end_to_end_net, init_view, trans, het, device, rot_mat_axis=rot_mat_axis, scale=init_scale)
    top_candidates, err, top_idx = keep_top_candidates(base_grid, base_errors, nb_kept)
    mean_errors.append(err.mean())
    
    if plot:
        plot_search_round(base_grid, base_errors, top_idx, 0, save_fold, index)
    if verbose: 
        print(f"Base grid was: \n{base_grid.shape}\n{base_grid}")
        print(f"with errors: \n{base_errors.shape}\n{base_errors}")
        print("\n")
        print(f"Top candidates found: \n{top_candidates.shape}\n{top_candidates}")
        print(f"with errors: \n{err.shape}\n{err}")

    # boucle d'itérations
    sigma = sigma_start
    for iter in range(max_iter):
        if scale_factors:
            iter_scale = scale_factors[iter + 1]
            iter_view = rescale_volume(view, scale_factor=iter_scale)
            print(f"View downsampled to {iter_view.shape}")
            save(f'{params_learning_setup["save_fold"]}/downsampled_{iter + 1}.tiff', to_numpy(iter_view))
        else:
            iter_scale = None
            iter_view = view

        flattened, sigma = subdivide_candidates(top_candidates, sigma, sigma_shrink)
        children_grid = build_children_rotation_grid(flattened, sigma, nb_children, device)
        children_grid = children_grid.reshape(nb_kept * nb_children, 3, 3)
        children_errors = evaluate_candidates(children_grid, end_to_end_net, iter_view, trans, het, device, rot_mat_axis=rot_mat_axis, scale=iter_scale)
        if iter+1 != max_iter: 
            # Normal iteration, keep nb_kept best
            top_candidates, err, top_idx = keep_top_candidates(children_grid, children_errors, nb_kept)
            top_candidates = top_candidates.reshape(bs, nb_kept, 3, 3)
        else:
            # Last iteration, keep 1 best
            top_candidates, err, top_idx = keep_top_candidates(children_grid, children_errors, 1)
            top_candidates = top_candidates.reshape(bs, 1, 3, 3)

        mean_errors.append(err.mean())

        if plot:
            plot_search_round(children_grid,  children_errors,  top_idx, iter+1,  save_fold, index)
        if verbose: 
            print(f"Children grid was: \n{children_grid.shape}\n{children_grid}")
            print(f"with errors: \n{children_errors.shape}\n{children_errors}")
            print("\n")
            print(f"Top candidates found: \n{top_candidates.shape}\n{top_candidates}")
            print(f"with errors: \n{err.shape}\n{err}")

    return top_candidates, mean_errors

def build_base_rotation_grid(nb_view_dir, 
                             nb_in_plane,
                             convention,
                             device):
    """
    Génère des rotations uniformément réparties dans l'espace des rotations SO(3)
    - nb_view_dir: nombre de directions de vue (theta, phi)
    - nb_in_plane: nombre de rotations in-plane (psi)
    - convention: convention d'Euler utilisée pour générer les matrices de rotation
    """
    theta, phi, psi = discretize_sphere_uniformly(nb_view_dir, nb_in_plane)
    euler_grid = np.array([[theta[i], phi[i], s] for i in range(len(theta)) for s in psi])
    rot_mats = get_3d_rotation_matrix(euler_grid,  convention=convention)
    return torch.tensor(rot_mats).float().cuda(device)

def build_children_rotation_grid(rot_mats, 
                                 sigma_deg, 
                                 nb_children,
                                 device):
    """
    Génère des rotations autour de rotations candidates avec une certaine dispersion angulaire
    - rot_mats: matrices de rotations candidates autour desquelles générer les enfants
    - sigma_deg: dispersion angulaire en degrés
    - nb_children: nombre d'enfants à générer pour chaque rotation candidate
    """
    nb_kept = rot_mats.shape[0]
    axis = torch.randn(nb_kept, nb_children, 3, device=device)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    angle = (sigma_deg * np.pi / 180) * torch.randn(nb_kept, nb_children, 1, device=device)
    axis_angle = (axis * angle).reshape(nb_kept * nb_children, 3)
    jitter = axis_angle_to_matrix(axis_angle).reshape(nb_kept, nb_children, 3, 3)
    return torch.einsum('klij,kjm->klim', jitter, rot_mats)

def evaluate_candidates(rot_mats,
                        end_to_end_net,
                        view,
                        trans,
                        het,
                        device,
                        rot_mat_axis=None,
                        scale=None):
    """
    Evalue les erreurs pour un ensemble de matrices de rotation candidates en comparant la vue générée avec la vue réelle
    - rot_mats: matrices de rotation candidates à évaluer
    - end_to_end_net: architecture end to end de Fluofire
    - view: vue réelle à comparer
    - trans: vecteur de translation associé
    - het: paramètre de conformation associé
    - scale: facteur d'échelle pour la vue si le sous-échantillonnage progressif est utilisé
    """
    bs = view.shape[0]
    nb_candidates = rot_mats.shape[0] 
    errors = torch.empty(bs, nb_candidates, device=view.device)
    for idx in range(nb_candidates):
        rot_mat = rot_mats[idx].unsqueeze(0).reshape(bs, 3, 3)
        errors[:, idx] = compute_errors(end_to_end_net, view, trans, het, rot_mat, device, rot_mat_axis=rot_mat_axis, scale=scale)
    return errors

@torch.no_grad()
def compute_errors(end_to_end_net,
                   view,
                   trans,
                   het,
                   rot_mat,
                   device,
                   rot_mat_axis=None,
                   scale=None,
                   test=False):
    """
    Evalue l'erreur pour une matrice de rotation candidate en comparant la vue générée avec la vue réelle
    - end_to_end_net: architecture end to end de Fluofire
    - view: vue réelle à comparer
    - trans: vecteur de translation associé
    - het: paramètre de conformation associé
    - rot_mat: matrice de rotation candidate à évaluer
    - scale: facteur d'échelle pour la vue si le sous-échantillonnage progressif est utilisé
    """
    bs = rot_mat.shape[0]    
    if not test: 
        # Use forward model to compare out volume with generated pose
        out_vol, _ = end_to_end_net.forward_decoder(rot_mat, trans, het, True, rot_mat_axis, scale=scale)
        out_conv = end_to_end_net.psf_layer.forward(out_vol)
        err = ((out_conv - view) ** 2).to(device).mean()
    else: 
        # Use ground truth and rotate to compare with generated pose (only for testing)
        gt_path = f"{PATH_PROJECT_FOLDER}/ground_truths/recepteurs_AMPA.tif"
        ground_truth = io.imread(gt_path)
        rotated, _ = rotation(to_numpy(view).squeeze(), to_numpy(rot_mat).squeeze().T)
        err = torch.from_numpy(((rotated - ground_truth) ** 2)).to(device).mean()
    return err.reshape(bs, 1)

def keep_top_candidates(rot_mats,
                        errors,
                        nb_kept):
    """
    Garde les meilleures matrices de rotation candidates en fonction des erreurs associées
    - rot_mats: matrices de rotation candidates à évaluer
    - errors: erreurs associées à chaque matrice de rotation candidate
    - nb_kept: nombre de meilleures matrices de rotation à conserver
    """
    topk_err, topk_idx = torch.topk(errors, nb_kept, dim=1, largest=False)
    top_candidates = rot_mats[topk_idx]   
    top_candidates_err = topk_err   
    return top_candidates, top_candidates_err, topk_idx

def subdivide_candidates(rot_mats,
                         sigma,
                         sigma_shrink):
    """
    Reshape les matrices de rotation candidates et ajuste la dispersion angulaire pour la génération des enfants de la prochaine itération
    - rot_mats: matrices de rotation candidates gardées après l'évaluation
    - sigma: dispersion angulaire actuelle
    - sigma_shrink: facteur de réduction de la dispersion angulaire
    """
    bs, nb_kept = rot_mats.shape[:2]
    flattened = rot_mats.reshape(bs * nb_kept, 3, 3)
    new_sigma = sigma / sigma_shrink
    return flattened, new_sigma