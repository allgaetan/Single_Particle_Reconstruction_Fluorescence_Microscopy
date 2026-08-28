from reconstruction_with_dl.SIREN import *
from reconstruction_with_dl.pose_net import to_numpy
from reconstruction_with_dl.losses import L2LossPreFlip
from reconstruction_with_dl.pose_net import PosesPredictor, rot_rep_dict
from reconstruction_with_dl.donut_cylinder import CylinderDecoder
from reconstruction_with_dl.data_set_views import get_mgrid, from_numpy_float32, ViewsRandomlyOrientedSimData
import os
from volume_representation.gaussian_mixture_representation.GMM_grid_evaluation import make_grid, nd_gaussian
from manage_matplotlib.graph_setup import set_up_graph
from common_image_processing_methods.rotation_translation import (get_rot_vec_from_rot_mat, get_3d_rotation_matrix,
                                                                   get_rotation_matrix, rotation)
from common_image_processing_methods.others import resize
from common_image_processing_methods.registration import registration_exhaustive_search
import matplotlib.pyplot as plt
from manage_files.read_save_files import (write_array_csv, save_pickle, read_images_in_folder, save_4d_for_chimera,
                                          read_csv, save, make_dir, read_image, print_dictionnary_in_file, load_pickle)
import tifffile
import matplotlib as mpl

from manage_files.paths import PATH_PROJECT_FOLDER

from stage_gaetan.pose_search import PoseTable, ConfTable

class PSF_layer(nn.Module):
    def __init__(self, psf_numpy, sidelenght, nb_dim, device, fourier=False):
        super().__init__()
        self.sidelenght = sidelenght
        self.nb_dim = nb_dim
        self.fourier = fourier
        if psf_numpy is not None:
            self.psf = from_numpy_float32(psf_numpy)
            self.psf_fourier = torch.fft.fftn(torch.fft.ifftshift(self.psf)).cuda(device)
        else:
            self.psf = None

    def forward(self, x):
        if self.psf is None:
            return x
        else:
            if self.fourier: x_convolved = self.psf_fourier * x
            else:
                x_convolved = torch.real(torch.fft.ifftn(self.psf_fourier * torch.fft.fftn(x)))
            return x_convolved

def freeze_net(net):
    for para in net.parameters():
        para.requires_grad = False


def unfreeze_net(net):
    for para in net.parameters():
        para.requires_grad = True


class End_to_end_architecture_volume(nn.Module):
    def __init__(self, params_learn_setup, params_data_gen) -> None:
        """modèle d'apprentissage de bout en bout composé d'un encodeur qui estime de manière non amorti les poses et le paramètre
        d'hétérogénéité et d'un décodeur, composé d'un opérateur de rotation et translation d'une grille 3d, d'une représentation
        implicite du volume sous forme de SIREN et de la convolution par la PSF
        L'initialisation de l'objet se fait à partir des dictionnaires params_learn_setup et params_data_gen qui regroupent un grand
        nombre de paramètres. Les valeurs par défauts des paramètres des dictionnaires sont dans le fichier default_params.py (dossier test_params).
    """
        super().__init__()
        assert params_learn_setup["rot_representation"] in rot_rep_dict.keys()
        self.fourier = params_learn_setup["fourier"]
        im_size = params_data_gen["size"]
        nb_dim = params_data_gen["nb_dim"]
        device = params_learn_setup["device"]
        self.pose_net = PosesPredictor(im_size, params_learn_setup, nb_dim)
        self.pose_table = PoseTable(params_learn_setup, params_data_gen)
        self.conf_table = ConfTable(params_learn_setup, params_data_gen)
        nb_dim_siren = nb_dim if not params_learn_setup["heterogeneity"] else nb_dim + params_learn_setup["nb_dim_het"]
        self.nb_channels = params_learn_setup["nb_channels"]
        self.img_siren = Siren(nb_dim_siren, out_features=self.nb_channels, first_omega_0=params_learn_setup["omega"], hidden_omega_0=params_learn_setup["omega"])
        self.nb_dim = nb_dim
        self.grid_unflatten, self.reference_grid = get_mgrid(im_size, nb_dim)
        self.grid_step = 2 / im_size
        self.device = device
        self.grid_unflatten = self.grid_unflatten.cuda(device)
        self.reference_grid = self.reference_grid.cuda(device)
        self.cylinder_decoder = CylinderDecoder(self.grid_unflatten, params_learn_setup, im_size)
        self.impose_cylinder = params_learn_setup["impose_cylinder"]
        self.unflatten_layer = nn.Unflatten(-2, tuple([im_size] * self.nb_dim))
        self.params = params_learn_setup
        self.params_gen = params_data_gen
        self.psf = get_psf(params_data_gen)
        self.psf_layer = PSF_layer(self.psf, im_size, nb_dim, device, self.fourier)
        # self.reference_grid.requires_grad = True
        self.est_rot_mat_0 = torch.eye(nb_dim).cuda(device)
        self.est_trans_0 = torch.zeros(nb_dim).cuda(device)
        self.size = im_size
        # self.gt = np.array(gt)end_to_end_net
        """
        if self.params["compute_metric"]:
            gt = []
            for c in range(self.nb_channels):
                gt_c = read_4d(f'{params_data_gen["pth_views"]}/c{c+1}/gt/4d_vol_channel.tiff')
                gt.append(gt_c)
            self.gt = np.array(gt)
        """

    def freeze_pose_table(self):
        freeze_net(self.pose_table)

    def freeze_conf_table(self):
        freeze_net(self.conf_table)

    def freeze_decoder(self):
        freeze_net(self.img_siren)

    def freeze_encoder(self):
        freeze_net(self.pose_net)

    def unfreeze_pose_table(self):
        unfreeze_net(self.pose_table)

    def unfreeze_conf_table(self):
        unfreeze_net(self.conf_table)

    def unfreeze_decoder(self):
        unfreeze_net(self.img_siren)

    def unfreeze_encoder(self):
        unfreeze_net(self.pose_net)

    def init_siren(self, init_img):
        self.img_siren, init_img, _ = fit_SIREN_with_image(init_img, self.params)
        return init_img

    def duplicated_grid_by_rotating(self, reference_grid, rot_mat_axis, rot_mat_view, symmetry_c=9):
        rotated_grids = []
        rot_mat_view = to_numpy(rot_mat_view)
        for c in range(symmetry_c):
            rot_mat_sym = get_rotation_matrix([0, 0, 360 * c / 9], 'XYZ')
            big_rot_mat = rot_mat_view @ np.linalg.inv(rot_mat_axis) @ rot_mat_sym @ rot_mat_axis # @ np.linalg.inv(rot_mat_view)
            big_rot_mat = torch.FloatTensor(big_rot_mat).cuda(self.device)
            rotated_grid = torch.matmul(reference_grid, big_rot_mat) # a checker !!
            rotated_grids.append(rotated_grid.unsqueeze(0))
        rotated_grids = torch.cat(rotated_grids)
        rotated_grids = torch.transpose(rotated_grids, 0, 1)
        return rotated_grids

    def concatenated_grid_and_het(self, grid, est_heterogeneite, rot_mat_axis=None):
        # est_heterogeneite += self.params["vae_param"] * torch.randn(est_heterogeneite.shape).cuda(self.device)
        if rot_mat_axis is None:
            est_heterogeneite_repeated = est_heterogeneite.unsqueeze(1).repeat(1, grid.shape[1], 1)
            concat_grid_heterogeneity = torch.cat((grid, est_heterogeneite_repeated), 2)
        else:
            est_heterogeneite_repeated = est_heterogeneite.unsqueeze(1).unsqueeze(1).repeat(1, grid.shape[1],
                                                                                            grid.shape[2], 1)
            concat_grid_heterogeneity = torch.cat((grid, est_heterogeneite_repeated), 3)
        return concat_grid_heterogeneity

    def translate_and_rotate_grid(self, rot_mat, trans, rot_mat_axis=None):
        reference_grid = self.reference_grid
        reference_grid = reference_grid.unsqueeze(0).repeat(trans.shape[0], 1, 1)
        trans_repeated = trans.unsqueeze(1).repeat(1, reference_grid.shape[1], 1)
        reference_grid = reference_grid - trans_repeated
        if rot_mat_axis is None:
            rotated_grid = torch.matmul(reference_grid, rot_mat)
        else:
            rotated_grid = self.duplicated_grid_by_rotating(reference_grid, rot_mat_axis, rot_mat)
        return rotated_grid

    def save_model(self, save_fold):
        """fonction permettant de sauvegarder le modèle pour une utilisation future. Le modèle pourra être lue par la suite avec la fonction
        load_pickle dans le dossier read_save_files
        """
        save_pickle(self, save_fold, 'saved_model')

    def unflatten(self, out_siren):
        out_siren_reshaped = self.unflatten_layer(out_siren)
        if len(out_siren_reshaped.shape) == 5:
            out_siren_reshaped = torch.permute(out_siren_reshaped, (0, 4, 1, 2, 3))
        else:
            out_siren_reshaped = torch.permute(out_siren_reshaped, (0, 5, 1, 2, 3,4))
        return out_siren_reshaped

    def forward_decoder(self, rot_mat, trans, est_heterogeneite, test=False, rot_mat_axis=None, scale=None):
        """forward pass à travers le décodeur. Fonction composée de deux blocs : un premier bloc de rotation et translation
        de la grille 3d et un bloc de passage de la grille à travers le SIREN.
        L'option 'impose_cylinder' permet de spécifier que l'on veut utiliser la représentation sosu forme de tor plutot que la
        représentation SIREN. Lorsque l'on veut imposer la symetri c9 à la reconstruction, la matrice de rotation qqui permet de passer
        de l'axe z à l'axe de symétrie du cylindre doit être spécifiée (variable rot_mat_axis). """
        if scale:
            scaled_size = int(self.size * scale)
            self.grid_unflatten, self.reference_grid = get_mgrid(scaled_size, self.nb_dim)
            self.grid_unflatten = self.grid_unflatten.cuda(self.device)
            self.reference_grid = self.reference_grid.cuda(self.device)
            self.unflatten_layer = nn.Unflatten(-2, tuple([scaled_size] * self.nb_dim))
            #self.psf = resize(self.psf, (scaled_size, scaled_size, scaled_size))
            self.params_gen["size"] = scaled_size
            self.psf = get_psf(self.params_gen)
            self.psf_layer = PSF_layer(self.psf, scaled_size, self.nb_dim, self.device, self.fourier)

        rotated_grid = self.translate_and_rotate_grid(rot_mat, trans, rot_mat_axis)
        if not self.impose_cylinder:
            if self.params["heterogeneity"]:
                concat_grid_heterogeneity = self.concatenated_grid_and_het(rotated_grid, est_heterogeneite, rot_mat_axis)
            else:
                concat_grid_heterogeneity = rotated_grid
            out_siren, _ = self.img_siren(concat_grid_heterogeneity)
            out_vol_reshaped = self.unflatten(out_siren)
            additional_regul = 0
            # out shape : (bs, Nc, s, s, s)
        else:
            out_vol_reshaped, additional_regul = self.cylinder_decoder.forward(est_heterogeneite, rotated_grid)
        if self.fourier:
            out_vol_ifftn = torch.real(torch.fft.ifftn(out_vol_reshaped))

        else: out_vol_ifftn = None
        return out_vol_reshaped, additional_regul, out_vol_ifftn

    def forward(self, view, t_vec, true_rot_mat, rot_mat_axis=None, test=False, pass_het=True, known_rot=False, known_trans=False):
        """takes a mini batch of input views and pass it through the encoder-decoder system, retturns the estimated view,
        translation vector and rotation matrix associated to the input view
        true_rot_mat and t_vec are used iff the rotation or translation (respectively) are supposed to be knwon"""
        est_rot_mat, est_trans, est_heterogeneite = self.pose_net.forward(view, test, pass_het=pass_het, known_rot=known_rot, known_trans=known_trans) # forward pass through the encoder
        rot_mat = true_rot_mat if known_rot else est_rot_mat
        trans = t_vec if known_trans else est_trans
        # reference_grid : N, 3
        out_siren_reshaped, additional_regul, out_siren_spatial = self.forward_decoder(rot_mat, trans, est_heterogeneite, test, rot_mat_axis) #forward pass through the decoder (except the psf)
        if not test:
            out_siren_convolved = self.psf_layer.forward(out_siren_reshaped) #forward pass throught the psf layer
        else:
            out_siren_convolved = out_siren_reshaped
        return out_siren_convolved, out_siren_reshaped, trans, rot_mat, est_rot_mat, est_heterogeneite, self.img_siren, additional_regul

    """
    def find_rot_mat_btw_centriole_and_z_axis(self):
        if self.params.heterogeneity:
            heterogeneity = torch.zeros(self.reference_grid.shape).cuda(0)
            concat_grid_heterogeneity = torch.cat((self.reference_grid, heterogeneity))model_output_convolved
        else:
            concat_grid_heterogeneity = self.reference_grid
        out_siren, _ = self.img_siren(concat_grid_heterogeneity)
        out_siren = to_numpy(out_siren).reshape([self.size] * self.nb_dim)
        rot_mat_axis = find_rot_mat_between_centriole_axis_and_z_axis(out_siren)
        return rot_mat_axis
    """

    def forward_rotation(self, random_rot_mat, random_trans_vec, heterogeneity_val, params_learn_setup):
        """inspired from paper ACE-HetEM : try to stabilize the learning process by fitting random inpt poses"""
        siren_eval, _, _ = self.forward_decoder(random_rot_mat, random_trans_vec, heterogeneity_val)
        out_siren_convolved = self.psf_layer.forward(siren_eval)
        est_rot_mat, est_trans, _ = self.pose_net.forward(out_siren_convolved, known_rot=params_learn_setup["known_rot"], known_trans=params_learn_setup["known_trans"])
        est_rot_mat = random_rot_mat if params_learn_setup["known_rot"] else est_rot_mat
        est_trans = random_trans_vec if params_learn_setup["known_trans"] else est_trans
        return est_rot_mat, est_trans

    def augment_batch(self, input, sym_loss):
        """
        Augment the dataset (batch-zise) with flipped images.

        Parameters
        ----------
        3d
        input: torch.Tensor (B, Nc, S, S, S)

        Returns
        -------
        out: torch.Tensor (4*B, Nc, S, S, S)
        2d :
        input: torch.Tensor (B, 1, S, S)
        out: torch.Tensor (2*B, 1, S, S)
        """
        if sym_loss:
            if self.nb_dim == 3:
                out = torch.cat((input, torch.flip(input, [3, 4]), torch.flip(input, [2,3]), torch.flip(input, [2,4])), 0)
            else:
                out = torch.cat((input, torch.flip(input, [2, 3])), 0)
        else:
            out = input
        return out

    def register_recons_4d(self, est_vol, true_vol, save_fold, save_name, rot_mat=None):
        """est vol : 4d array shape (nb views, S, S, S)  (heterogene reconstruction)
        true vol : ground truth object
        registere est vol on true vol"""
        id = est_vol.shape[0]-3
        est_vol_middle = est_vol[id, :,:,:]
        true_vol_middle = true_vol[id, :,:,:]
        if rot_mat is None:
            print('is none')
            rot_vec, _ = registration_exhaustive_search(true_vol_middle, est_vol_middle, '', '', 3, save_res=False, sample_per_axis=40)
            print('rot vec', rot_vec)
            rot_mat = get_3d_rotation_matrix(np.degrees(rot_vec), convention='ZYX')
        print('rot mat', rot_mat)
        registered_est_vol = []
        for t in range(len(est_vol)):
            rotated, _ = rotation(est_vol[t], rot_mat)
            registered_est_vol.append(rotated)
        registered_est_vol = np.array(registered_est_vol)
        save_4d_for_chimera(registered_est_vol, f'{save_fold}/{save_name}.tiff')
        return registered_est_vol, rot_mat

    def checkpoint(self, data_set, save_fold, params_data_gen, params_learn_setup):
        device, convention_data_gen = params_learn_setup["device"], params_data_gen["convention"]
        nb_views = data_set.nb_views
        true_rot_vecs = []
        est_rot_vecs = []
        true_heterogeneities = []
        est_heterogeneities = []
        est_vols = []
        est_transs =[]
        true_rot_mats = []
        est_rot_mats = []
        file_names = []
        views = []
        view_0 = data_set[0][1].unsqueeze(0).cuda(device)
        index_0 = data_set[0][0]
        est_rot_mat_0, est_trans_0 = self.pose_table.forward_sgd(index_0)
        est_rot_mat_0 = est_rot_mat_0.cuda(device)
        est_trans_0 = est_trans_0.cuda(device)
        print('est rot mat 0 shape', est_rot_mat_0.shape)
        self.est_rot_mat_0 = est_rot_mat_0
        self.est_trans_0 = est_trans_0
        #self.save_model(save_fold)
        fold_save_align = f'{save_fold}/realigned_views'
        #make_dir(fold_save_align)
        for v, d in enumerate(data_set):
            index, view, _, rot_mat, rot_vec, t_vec, dilatation_val, file_name = d
            file_names.append(file_name)
            index = index
            view = view.cuda(device)
            rot_mat = rot_mat.cuda(device)
            t_vec = t_vec.unsqueeze(0).cuda(device)

            est_heterogeneity = self.conf_table.forward_sgd(index)
            est_rot_mat, est_trans = self.pose_table.forward_sgd(index)

            est_vol, _, est_vol_ifftn = self.forward_decoder(est_rot_mat, est_trans, est_heterogeneity, True)
            if self.fourier: est_vol = est_vol_ifftn

            if v < 0:
                # Realign views with predicted poses to check if the estimated poses are relatively correct
                reverse, _ = rotation(to_numpy(view).squeeze(), to_numpy(est_rot_mat).squeeze().T)
                reverse = np.array(reverse)
                save(f'{fold_save_align}/views_{v}.tiff', reverse)

            views.append(to_numpy(view))
            if est_rot_mat is not None:
                est_rot = get_rot_vec_from_rot_mat(to_numpy(est_rot_mat).squeeze(), convention_data_gen)
                est_rot_mats.append(to_numpy(est_rot_mat.squeeze()))
                est_rot_vecs.append(est_rot % 360)

            if est_trans is not None:
                est_transs.append(to_numpy(est_trans).squeeze())

            if est_heterogeneity is not None:
                est_heterogeneity = to_numpy(est_heterogeneity)
                est_heterogeneities.append(est_heterogeneity)
            true_rot_vecs.append((to_numpy(rot_vec) * 180 / np.pi) % 360)
            est_vols.append(to_numpy(est_vol))
            true_rot_mats.append(to_numpy(rot_mat))
            true_heterogeneities.append(dilatation_val)

        est_rot_mats = np.array(est_rot_mats)
        true_rot_vecs = np.array(true_rot_vecs)
        est_rot_vecs = np.array(est_rot_vecs)
        true_rot_mats = np.array(true_rot_mats)
        true_heterogeneities = np.array(true_heterogeneities)
        est_heterogeneities = np.array(est_heterogeneities)
        est_heterogeneities = est_heterogeneities.squeeze()
        if len(est_rot_vecs) > 0:
            write_array_csv(true_rot_vecs, f'{save_fold}/true_rots.csv')
            write_array_csv(est_rot_vecs, f'{save_fold}/est_rots.csv')
        fold_save_vol = f'{save_fold}/vols'
        make_dir(fold_save_vol)
        save(f"{fold_save_vol}/recons.tiff", est_vols[0])
        return est_heterogeneities, est_rot_mats, est_transs, file_names, est_vols, views

    def curent_inference(self, data_set, save_fold, params_data_gen, params_learn_setup):
        device, convention_data_gen = params_learn_setup["device"], params_data_gen["convention"]
        nb_views = data_set.nb_views
        true_rot_vecs = []
        est_rot_vecs = []
        true_heterogeneities = []
        est_heterogeneities = []
        est_vols = []
        est_transs =[]
        true_rot_mats = []
        est_rot_mats = []
        file_names = []
        views = []
        angle_errors = []
        deltas = []
        view_0 = data_set[0][1].unsqueeze(0).cuda(device)
        est_rot_mat_0, est_trans_0, _ = self.pose_net.forward(view_0, test=True)
        est_rot_mat_0 = est_rot_mat_0.cuda(device)
        est_trans_0 = est_trans_0.cuda(device)
        print('est rot mat 0 shape', est_rot_mat_0.shape)
        self.est_rot_mat_0 = est_rot_mat_0
        self.est_trans_0 = est_trans_0
        #self.save_model(save_fold)
        fold_save_align = f'{save_fold}/realigned_views'
        #make_dir(fold_save_align)
        for v, d in enumerate(data_set):
            #print(d)
            _, view, _, rot_mat, rot_vec, t_vec, dilatation_val, file_name = d
            file_names.append(file_name)
            view = view.cuda(device)
            rot_mat = rot_mat.cuda(device)
            t_vec = t_vec.unsqueeze(0).cuda(device)
            _, est_vol, est_trans, est_rot_mat, _, est_heterogeneity, _, _ = self.forward(view.unsqueeze(0), t_vec,
                                                                 rot_mat, test=True, known_rot=params_learn_setup["known_rot"], known_trans=params_learn_setup["known_trans"])
            if self.impose_cylinder:
                """if cylinder save cylinder parametres, which are more interpretable than the heterogeneity param"""
                if self.nb_channels == 1:
                    lenght, radius, width = self.cylinder_decoder.get_cylinder_param_from_het_value(est_heterogeneity)
                    est_heterogeneity = torch.tensor([lenght, radius, width])
                else:
                    lenght, radius, width, pos = self.cylinder_decoder.get_cylinder_param_from_het_value_2_channels(est_heterogeneity)
                    lenght, radius, width, pos = lenght.squeeze(), radius.squeeze(), width.squeeze(), pos.squeeze()
                    est_heterogeneity = torch.tensor([lenght[0], lenght[1], radius[0], radius[1], width[0], width[1] , pos])
                #est_pos = self.cylinder_decoder.get_pos_from_pos_param()
                #write_array_csv(to_numpy(est_pos), f'{save_fold}/est_pos_cylinder_2.csv')
            
            if v < 0:
                reverse, _ = rotation(to_numpy(view).squeeze(), to_numpy(est_rot_mat).squeeze().T)
                reverse = np.array(reverse)
                save(f'{fold_save_align}/views_{v}.tiff', reverse)

            views.append(to_numpy(view))
            if est_rot_mat is not None:
                est_rot = get_rot_vec_from_rot_mat(to_numpy(est_rot_mat).squeeze(), convention_data_gen)
                est_rot_mats.append(to_numpy(est_rot_mat.squeeze()))
                est_rot_vecs.append(est_rot % 360)# * np.pi / 180)

                delta = to_numpy(est_rot_mat.squeeze()) @ to_numpy(rot_mat).T
                deltas.append(np.asarray(delta).reshape(-1))
                angle_error = np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1))
                angle_errors.append(angle_error)

            if est_trans is not None:
                est_transs.append(to_numpy(est_trans).squeeze())

            if est_heterogeneity is not None:
                est_heterogeneity = to_numpy(est_heterogeneity)
                est_heterogeneities.append(est_heterogeneity)
            true_rot_vecs.append((to_numpy(rot_vec) * 180 / np.pi) % 360)
            est_vols.append(to_numpy(est_vol))
            true_rot_mats.append(to_numpy(rot_mat))
            true_heterogeneities.append(dilatation_val)

        est_rot_mats = np.array(est_rot_mats)
        true_rot_vecs = np.array(true_rot_vecs)
        est_rot_vecs = np.array(est_rot_vecs)
        angle_errors = np.array(angle_errors)
        true_rot_mats = np.array(true_rot_mats)
        deltas = np.array(deltas)
        true_heterogeneities = np.array(true_heterogeneities)
        est_heterogeneities = np.array(est_heterogeneities)
        est_heterogeneities = est_heterogeneities.squeeze()
        if len(est_rot_vecs) > 0:
            write_array_csv(true_rot_vecs, f'{save_fold}/true_rots.csv')
            write_array_csv(est_rot_vecs, f'{save_fold}/est_rots.csv')
            write_array_csv(angle_errors, f'{save_fold}/angle_errors.csv')
            write_array_csv(deltas, f'{save_fold}/deltas.csv')
        fold_save_vol = f'{save_fold}/vols'
        make_dir(fold_save_vol)
        if self.params["heterogeneity"]:
            est_heterogeneities = est_heterogeneities.squeeze()
            write_array_csv(true_heterogeneities, f'{save_fold}/true_heterogeneities.csv')
            write_array_csv(est_heterogeneities, f'{save_fold}/est_heterogeneities.csv')
            if len(est_heterogeneities.shape) == 1:
                est_heterogeneities = np.expand_dims(est_heterogeneities, 1)
            if len(true_heterogeneities.shape) == 1:
                true_heterogeneities = np.expand_dims(true_heterogeneities, 1)

            plot_heterogeneity(true_heterogeneities, est_heterogeneities, save_fold)
            if params_learn_setup["nb_dim_het"] == 2:
                scatter_two_dim_het(est_heterogeneities, true_heterogeneities, save_fold)
            est_het_order = np.argsort(est_heterogeneities[:, 0])
            est_heterogeneities_sorted = est_heterogeneities[est_het_order]
            write_array_csv(est_heterogeneities_sorted, f'{save_fold}/est_heterogeneities_sorted.csv')
            est_vols = np.array(est_vols)
            vols_ordered = est_vols[est_het_order]
            #for i in range(3):
            file_names_with_idx = np.array([range(nb_views), data_set.file_names]).T
            write_array_csv(file_names_with_idx, f'{save_fold}/idx_views.csv')
            file_names_sorted = np.array(data_set.file_names)[est_het_order]
            write_array_csv(np.array([file_names_sorted]).T, f'{save_fold}/file_names_sorted.csv')
            for c in range(self.nb_channels):
                if len(vols_ordered.shape) == 5:
                    vols_ordered = np.expand_dims(vols_ordered, axis=1)
                vols_channel = vols_ordered[:,:,c,:,:,:] # shape (nb_views, 1, s, s, s)
                vols_channel = np.transpose(vols_channel, (0,2,1,3,4)) # shape (nb_views, s, 1, s, s)
                tifffile.imwrite(f'{fold_save_vol}/4d_vol_channel_{c+1}.tiff', vols_channel, imagej=True)
        else:
            save(f'{fold_save_vol}/recons.tiff', est_vols[0])
        return est_heterogeneities, est_rot_mats, est_transs, file_names, est_vols, views

    def predict_regulraly_spaced_het(self, est_heterogeneities):
        fold_final = f'{save_fold}/final_vols'
        make_dir(fold_final)
        #if len(est_heterogeneities.shape) == 1:
        ma = np.max(est_heterogeneities)
        mi = np.min(est_heterogeneities)
        std = np.std(est_heterogeneities)
        avg = np.mean(est_heterogeneities)
        mi_rg = max(mi, avg - 3 * std)
        ma_rg = min(ma, avg + 3 * std)
        all_vols = []
        for het_val in np.linspace(mi_rg, ma_rg, 500):
            het_val_tensor = torch.FloatTensor([[het_val]]).cuda(self.device)
            recons_het_val, _, _ = self.forward_decoder(self.est_rot_mat_0, self.est_trans_0, het_val_tensor, True)
            recons_het_val_numpy = to_numpy(recons_het_val)
            all_vols.append(recons_het_val_numpy)
            # save(f'{fold_final}/recons_het_val_{het_val}.tif', recons_het_val_numpy)
            all_vols = np.array(all_vols)
        for c in range(self.nb_channels):
            vols_channel = all_vols[:, :, c, :, :, :]
            vols_channel = np.transpose(vols_channel, (0, 2, 1, 3, 4))
            tifffile.imwrite(f'{fold_final}/4d_vol_regularly_spapced_het_channel_{c + 1}.tiff', vols_channel,
                             imagej=True)

    def visu_angle_est(self, true_rot_vecs, est_rot_vecs, save_fold):
        #print('est rot shape', np.array(est_rot).shape)
        cm = plt.cm.get_cmap('hsv')

        def plot(true, est, d=0):
            set_up_graph((30,30))
            cosines = np.cos(np.array(true))
            sines = np.sin(np.array(true))
            plt.scatter(cosines, sines, c=est, vmin=np.min(est), vmax=np.max(est), s=200, cmap=cm)
            cbar = plt.colorbar(fraction=0.05, pad=0.04, format='%.0e')
            cbar.set_label('estimated angle (radians)')
            plt.xlabel('cosine of true angle')
            plt.ylabel('sine of true angle')
            plt.savefig(f'{save_fold}/visu_est_angles_{d}.png')
            plt.close()

        if self.nb_dim == 2:
            plot(true_rot_vecs, est_rot_vecs)
        else:
            for d in range(3):
                plot(true_rot_vecs[:, d], est_rot_vecs[:, d], d)

def compute_relative_pose_error(est_rot_mats, true_rot_mats, nb_views):
    I = np.eye(3)
    total = 0.0
    for l in range(nb_views):
        Ml = est_rot_mats[l].T @ true_rot_mats[l]
        for m in range(nb_views):
            Mm = est_rot_mats[m].T @ true_rot_mats[m]
            diff = Ml[0] @ Mm[0].T - I
            total += np.linalg.norm(diff, ord='fro')
    return total / (nb_views * nb_views)

def plot_heterogeneity(true_heterogeneities, est_heterogeneities, save_fold):
    """Trace le paramèter d'hétérogénéité réel en fonction du paramètre d'hétérogénéité estimé"""
    for l in range(est_heterogeneities.shape[1]):
        true_het_l = true_heterogeneities if len(true_heterogeneities.shape) == 1 else true_heterogeneities[:, 0]
        est_het_l = est_heterogeneities[:, l]
        m = int(np.max(true_het_l) + 1)
        # mean_est_het = [np.mean(est_het_l[true_het_l == i]) for i in range(1, m)]
        set_up_graph()
        plt.scatter(true_het_l, est_het_l)
        # plt.plot(range(1, m), mean_est_het, color='r', marker='X')
        plt.xlabel("Paramètre d'hétérogénéité réel")
        plt.ylabel("Paramètre d'hétérogénéité estimé")
        plt.grid()
        plt.savefig(f'{save_fold}/est_het_wrt_true_het_{l}.png')
        plt.close()


def scatter_two_dim_het(est_heterogeneities, true_heterogeneities, save_fold):
    """ affiche les hétérogénéités estimées en fonction des hétérogénéité réelles lorque l'espace latent présente
            2 dimensions.
     est_heterogeneities et true_heterogeneities sont des array 2d de dimension (nb_vues, 2) (true_heterogeneities peut également avoir
     une dimension (nb_vues, 1) s'il ya un seul paramètre réel et 2 paramètres estimés)
     Les deux graphes tracées sont des scatters plot 2d qui contiennent l'ensemble des points estimées de l'espace latent,
     les abscicce corresponde à la première dimension de l'espace latent et les ordonnées à la seconde dimension.
     Le premier graphes présente un dégradé de couleur en fonction du premier paramètre d'hétérogénéité réel et le second un dégradé
     par rapport au second paramètre d'hétérogénéité réel
     """
    cmap = plt.cm.hot
    for l in range(true_heterogeneities.shape[1]):
        ma = np.max(true_heterogeneities[:, l])
        mi = np.min(true_heterogeneities[:, l])

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(40, 20))
        #set_up_graph()
        # Normalize the true heterogeneities for the colormap
        norm = mpl.colors.Normalize(vmin=mi, vmax=ma)

        # Create the scatter plot
        sc = ax.scatter(est_heterogeneities[:, 0], est_heterogeneities[:, 1],
                        c=true_heterogeneities[:, l], cmap=cmap, norm=norm)

        ax.set_xlabel("Première dimension espace latent")
        ax.set_ylabel("Seconde dimension espace latent")
        ax.grid(True)

        # Create the colorbar
        cbar = plt.colorbar(sc, ax=ax)
        if l == 1:
            cbar.set_label("Largeur du centriole")
        else:
            cbar.set_label("Longueur du centriole")

        # Save the figure
        plt.savefig(f'{save_fold}/est_het_two_dim_{l}')
        plt.close()
"""
def scatter_two_dim_het(est_heterogeneities, true_heterogeneities, save_fold):
    cmap = plb.cm.hot
    #nb_points = len(true_heterogeneities[:,0])
    for l in range(2):
        ma = np.max(true_heterogeneities[:,l])
        mi = np.min(true_heterogeneities[:,l])
        set_up_graph()
        for i in range(len(est_heterogeneities)):
            plt.scatter(est_heterogeneities[i,0], est_heterogeneities[i,1], color=cmap(int((true_heterogeneities[i,l] - mi)/(ma-mi) * cmap.N)))
        plt.xlabel("Première dimension espace latent")
        plt.ylabel("Seconde dimension espace latent")
        plt.grid()
        if l == 1:
            plt.colorbar(label="Longueur du centriole")
        else:
            plt.colorbar(label="Largeur du centriole")
        plt.savefig(f'{save_fold}/est_het_two_dim_{l}')
        plt.close()
"""

def get_psf(params_data_gen):
    if params_data_gen["psf"] is None:
        nb_dim = params_data_gen["nb_dim"]
        grid_step = 2/(params_data_gen["size"]-1)
        cov_PSF = grid_step ** 2 * np.eye(nb_dim)
        cov_PSF[0, 0] *= params_data_gen["sig_z"] ** 2
        for i in range(1, nb_dim):
            cov_PSF[i, i] *= params_data_gen["sig_xy"] ** 2
        grid = make_grid(params_data_gen["size"], nb_dim)
        psf = nd_gaussian(grid, np.zeros(nb_dim), cov_PSF, nb_dim)
        psf /= np.sum(psf)
        return psf
    else:
        return params_data_gen["psf"]/np.sum(params_data_gen["psf"])


def init_volume(end_to_end_net, params_learn_setup, size, views):
    """initialise la représentation implicite du volume. 3 facon d'initialiser : avec la moyenne des vues
    (params_learn_setup["init"] == 'avg_views'), initialisation aléatoire (params_learn_setup["init"] == 'random'),
    initialisation avec un volume init_vol spécifié ('params_learn_setup["init"] == init_vol, init_vol doit être un array 3d)"""
    save_fold = params_learn_setup["save_fold"]
    if params_learn_setup["init"] == 'avg_views':
        init_vol = np.mean(np.array(views), axis=0).squeeze()
    elif params_learn_setup["init"] == 'random':
        init_vol = None
    else:
        print('ici')
        init_vol = read_image(params_learn_setup["init"])
        init_vol = resize(init_vol, (size, size, size))
        save(f'{save_fold}/init_vol.tif', init_vol)
    if init_vol is not None:
        print('init siren')
        init_vol_fit = end_to_end_net.init_siren(init_vol)
        # save(f'{PATH_PROJECT_FOLDER}/test_init_siren/init_{nm}.tif', init_vol_fit)
        save(f'{save_fold}/init_vol_fit.tif', init_vol_fit)
        print('volume initialized')


def init_phases_pose_to_pose(params_learning_setup):
    """
    returns a list of boolean, phases, for the pose to pose training. Phases[ep] = True if ep is
    an epoch associated to an image to image training and is Falses if it is
    an epoch associated to a pose to pose training"""
    a = params_learning_setup["nb_epochs_each_phases_ACE_Het"]
    if a is not None:
        phases = []
        phase_im_to_im = True
        while len(phases) < params_learning_setup["nb_epochs"]:
            id = 0 if phase_im_to_im else 1
            phases = phases + [phase_im_to_im] * a[id]
            phase_im_to_im = not phase_im_to_im
        phases = phases[
                 :params_learning_setup["nb_epochs"]]
    else:
        phases = [True] * params_learning_setup["nb_epochs"]
    return phases

def init_phases_hps(params_learning_setup):
    hps = params_learning_setup["nb_epochs_hps"]
    if hps is not None:
        phases = [True] * hps + [False] * (params_learning_setup["nb_epochs"] - hps)
    else:
        phases = [True] * params_learning_setup["nb_epochs"]
    return phases

def save_parameters(params_learning_setup, params_data_gen, data_set):
    """sauvegarde l'ensemble des paramètres dans un fichier. Les 20 premières vues du jeu de données sont également sauvegardées"""
    save_fold = params_learning_setup["save_fold"]
    make_dir(save_fold)
    loc1 = f'{save_fold}/training_param_learn_setup.txt'
    loc2 = f'{save_fold}/datagen_param.txt'
    f1 = open(loc1, 'w')
    f2 = open(loc2, 'w')
    print_dictionnary_in_file(params_learning_setup, f1)
    print_dictionnary_in_file(params_data_gen, f2)
    save_pickle(params_data_gen, params_learning_setup["save_fold"], "params_data_gen")
    save_pickle(params_learning_setup, params_learning_setup["save_fold"], "params_learn_setup")

    psf = get_psf(params_data_gen)  # a recoder !!!
    save(f'{save_fold}/psf.tif', psf)
    for i, d in enumerate(data_set):
        make_dir(f'{save_fold}/views')
        if i <= 20:
            save(f'{save_fold}/views/view_{i}.tif', d[1].squeeze())


"""def train_bicanal_cylinder_rep(data_set, params_data_gen, params_learning_setup):
    if params_learning_setup["known_rot"]:
        assert not params_learning_setup["sym_loss"]
    if params_learning_setup["known_trans"] and params_learning_setup["sym_loss"]:
        assert params_data_gen["sigma_trans_ker"] == 0
    save_parameters(params_learning_setup, params_data_gen, data_set)
    dataloader = DataLoader(data_set, batch_size=params_learning_setup["bs"], pin_memory=True, num_workers=0)
    end_to_end_net = End_to_end_architecture_volume(params_learning_setup, params_data_gen, None)
    end_to_end_net.cuda(params_learning_setup["device"])
    size = params_data_gen["size"]
    optim = torch.optim.Adam(
        [{'params': end_to_end_net.pose_net.parameters(), 'lr': params_learning_setup["lr_pose_net"]}])
    losses = []
    coeff_reg_trans = params_learning_setup["init_coeff_reg_trans"]
    saved_het = []
    device = params_learning_setup["device"]
    print('ca commence')
    for ep in range(params_learning_setup["nb_epochs"]):
        total_loss = torch.zeros(1).cuda(params_learning_setup["device"])
        total_loss_regul_pos = torch.zeros(1).cuda(params_learning_setup["device"])
        h = 0
        bs = params_learning_setup["bs"]
        for v,d in enumerate(dataloader):
            _, _, two_cons_view, rot_mat, _, t_vec, _, _ = d
            t_vec = t_vec.cuda(device)
            two_cons_view = two_cons_view.cuda(device)
            two_cons_view = two_cons_view.squeeze(F, #)
            rot_mat = rot_mat.cuda(device)
            view_augmented = end_to_end_net.augment_batch(two_cons_view, params_learning_setup["sym_loss"])
            model_output_convolved, model_output, trans, est_rot_mat,_, est_het, siren_learned, additional_regul = end_to_end_net.forward(
                view_augmented, t_vec, rot_mat, None)

            loss = ((model_output_convolved - two_cons_view) ** 2).mean() * bs

            if not params_learning_setup["known_trans"] and params_learning_setup["use_reg_trans"]:
                coeff_reg_trans /= params_learning_setup["coeff_reg_trans_ratio"]
                loss += coeff_reg_trans * torch.sum(trans ** 2)
            radius_width1 = est_het[0][[2,4]]
            radius_width2 = est_het[1][[2,4]]
            regul_pos = torch.mean((radius_width2-radius_width1)**2)
            
            if params_learning_setup["heterogeneity"]:
                loss += 10 ** -4 * torch.mean(est_het ** 2)
            
            if not params_learning_setup["known_trans"] and params_learning_setup["use_reg_trans"]:
                coeff_reg_trans /= params_learning_setup["coeff_reg_trans_ratio"]
                loss += coeff_reg_trans * torch.sum(trans ** 2)
            coeff_regul_pos = params_learning_setup["coeff_regul_pos"]
            loss += coeff_regul_pos*regul_pos
            total_loss_regul_pos += coeff_regul_pos*regul_pos
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss
        avg_loss = to_numpy(total_loss / params_data_gen["nb_views"])
        avg_loss_regul_poss = to_numpy(total_loss_regul_pos / params_data_gen["nb_views"])
        print(f"Step {ep}, mean loss {avg_loss}, mean loss regul poss {coeff_regul_pos*avg_loss_regul_poss}")
        losses.append(avg_loss)
        x = params_learning_setup["x"]
        if ep % x == x - 1:
            # print('save to ', save_fold)
            inter_save_fold = f'{params_learning_setup["save_fold"]}/ep_{ep}'
            make_dir(inter_save_fold)
            end_to_end_net.curent_inference(data_set, inter_save_fold, params_data_gen, params_learning_setup)
    write_array_csv(np.array(losses), f'{save_fold}/losses.csv')"""

def train_with_hps(data_set, params_data_gen, params_learning_setup, gt_4d, views):
    """
    Only works in homogen with known translations for now
    """
    save_fold = params_learning_setup["save_fold"]
    dataloader = DataLoader(data_set, batch_size=params_learning_setup["bs"], pin_memory=True, num_workers=0, shuffle=True)
    #end_to_end_net = load_pickle(f"{PATH_PROJECT_FOLDER}/jobs/ampa_siren_trained/reconstruction/saved_model")
    end_to_end_net = End_to_end_architecture_volume(params_learning_setup, params_data_gen)
    end_to_end_net.cuda(params_learning_setup["device"])
    save_parameters(params_learning_setup, params_data_gen, data_set)

    size = params_data_gen["size"]
    coeff_reg_trans = params_learning_setup["init_coeff_reg_trans"]
    phases = init_phases_hps(params_learning_setup)
    device = params_learning_setup["device"]
    init_volume(end_to_end_net, params_learning_setup, size, views)
    optim = torch.optim.Adam([
        {'params': end_to_end_net.img_siren.parameters(), 'lr':params_learning_setup["lr_siren"]},
        {'params': end_to_end_net.pose_table.parameters(), 'lr':params_learning_setup["lr_pose_table"]},
        {'params': end_to_end_net.conf_table.parameters(), 'lr':params_learning_setup["lr_conf_table"]}
    ])
    losses = []
    rpes = []
    trans_abs_vals = []
    saved_het = []
    hps_mean_errors = []
    for ep in range(params_learning_setup["nb_epochs"]):      
        est_rot_mats = []
        true_rot_mats = []
        trans_abs_val = 0
        total_loss = torch.zeros(1).cuda(device)
        h = 0
        bs = params_learning_setup["bs"]
        hps_mean_errors_epoch = []
        for v, d in enumerate(dataloader):
            h+=1
            index, view, _, rot_mat, _, t_vec, _, _ = d 
            index = index.cuda(device)
            t_vec = t_vec.cuda(device)
            view = view.cuda(device)
            rot_mat = rot_mat.cuda(device)
            if end_to_end_net.fourier:
                view = torch.fft.fftn(view)

            het_val = torch.zeros((1, params_learning_setup["nb_dim_het"])).cuda(params_learning_setup["device"])
            end_to_end_net.conf_table.initialize(index, het_val)
            est_het = end_to_end_net.conf_table.forward_sgd(index)
            saved_het.append(est_het.detach())

            if phases[ep]:
                # HPS phase
                est_rot_mat, trans, hps_mean_errors_epoch_view = end_to_end_net.pose_table.forward_hps(
                    end_to_end_net,
                    view,
                    index.detach()[0],
                    est_het,
                    t_vec,
                    params_learning_setup,
                    params_data_gen,
                    true_rot_mat=rot_mat)
                hps_mean_errors_epoch.append([
                    to_numpy(err) for err in hps_mean_errors_epoch_view
                ])
            else:
                # SGD phase
                est_rot_mat, trans = end_to_end_net.pose_table.forward_sgd(index)

            # Loss
            est_rot_mats.append(to_numpy(est_rot_mat.squeeze()))  
            true_rot_mats.append(to_numpy(rot_mat))
            trans_abs_val += np.mean(np.abs(to_numpy(trans)))

            out_vol, _, _ = end_to_end_net.forward_decoder(est_rot_mat, 
                                                        trans, 
                                                        est_het, 
                                                        True)
            model_output_convolved = end_to_end_net.psf_layer.forward(out_vol)

            loss = torch.abs(model_output_convolved - view).square().mean()
            total_loss += loss
            optim.zero_grad()
            loss.backward()

            if phases[ep]:
                end_to_end_net.pose_table.table_rot.grad = None
                end_to_end_net.pose_table.table_trans.grad = None
            optim.step()

        rpe = compute_relative_pose_error(est_rot_mats, true_rot_mats, params_data_gen["nb_views"])
        rpes.append(rpe)

        avg_loss = to_numpy(total_loss/params_data_gen["nb_views"])
        if phases[ep]:
            print(f"Step {ep}, mean loss HPS phase {avg_loss}")
            hps_mean_errors_epoch = np.array(hps_mean_errors_epoch).transpose((1, 0)).mean(axis=1)
            hps_mean_errors.append(hps_mean_errors_epoch)
        else:
            print(f"Step {ep}, mean loss SGD phase {avg_loss}")
        print("trans abs val", trans_abs_val/h)
        trans_abs_vals.append(trans_abs_val/h)
        losses.append(avg_loss)
        x = params_learning_setup["x"]
        if phases[ep] or ep%x == x-1:
            inter_save_fold = f'{save_fold}/ep_{ep}'
            make_dir(inter_save_fold)

            plt.plot(losses)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss during reconstruction")
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{inter_save_fold}/losses_{ep}.png')
            plt.close()

            plt.plot(rpes)
            plt.xlabel("Epoch")
            plt.ylabel("Error")
            plt.title("Relative error between estimated poses and true poses")
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{inter_save_fold}/rpes_{ep}.png')
            plt.close()

            errors = np.array(hps_mean_errors).reshape(-1, params_learning_setup["hps_max_iter"] + 1)
            for i in range(params_learning_setup["hps_max_iter"] + 1):
                plt.plot(errors[:, i], label=f'{i}')
            plt.xlabel("Epoch")
            plt.ylabel("Mean error")
            plt.title("Mean errors of the pose estimation by HPS")
            plt.legend(title="Iteration of HPS")
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{inter_save_fold}/hps_mean_errors_{ep}.png')
            plt.close()

            end_to_end_net.checkpoint(data_set, inter_save_fold, params_data_gen, params_learning_setup)
    end_to_end_net.save_model(save_fold)

def train(data_set, params_data_gen, params_learning_setup, gt_4d, views):
    save_fold = params_learning_setup["save_fold"]
    if params_learning_setup["known_rot"]:
        assert not params_learning_setup["sym_loss"]
    if params_learning_setup["known_trans"] and params_learning_setup["sym_loss"]:
        assert params_data_gen["sigma_trans_ker"] == 0

    save_parameters(params_learning_setup, params_data_gen, data_set)
    dataloader = DataLoader(data_set, batch_size=params_learning_setup["bs"], pin_memory=True, num_workers=0, shuffle=False)
    end_to_end_net = End_to_end_architecture_volume(params_learning_setup, params_data_gen)
    #end_to_end_net = load_pickle(f"{PATH_PROJECT_FOLDER}/jobs/20260608_164907_0/reconstruction/saved_model")
    end_to_end_net.cuda(params_learning_setup["device"])
    size = params_data_gen["size"]
    init_volume(end_to_end_net, params_learning_setup, size, views)
    optim = torch.optim.Adam([{'params': end_to_end_net.img_siren.parameters(), 'lr':params_learning_setup["lr_siren"]},
                            {'params': end_to_end_net.pose_net.parameters(), 'lr': params_learning_setup["lr_pose_net"]}])
    sym_loss = L2LossPreFlip(params_data_gen["nb_dim"], params_learning_setup["loss_type"])
    losses = []
    losses_i2i = [np.array([0])]
    losses_p2p = [np.array([0])]
    
    rpes = []

    coeff_reg_trans = params_learning_setup["init_coeff_reg_trans"]
    trans_abs_vals = []
    rot_mat_axis = None #if not params_learning_setup["impose_symmetry"] else np.eye(3)
    phases = init_phases_pose_to_pose(params_learning_setup)
    saved_het = []
    device = params_learning_setup["device"]
    ep_start_imp_sym = 40 if not params_learning_setup["heterogeneity"] else 1400
    for ep in range(params_learning_setup["nb_epochs"]):

        est_rot_mats = []
        true_rot_mats = []

        if ep >=ep_start_imp_sym and params_learning_setup["impose_symmetry"]:
            rot_mat_axis = np.eye(3)
            params_learning_setup["x"] = 10
        """
        if ep == 1250 and params_learning_setup["impose_cylinder"]:
            end_to_end_net.pose_net.reinit_rot_params()
        """
        if phases[ep]:
            saved_het = []
        else:
            #saved_het = torch.zeros((params_data_gen["nb_views"], params_learning_setup["nb_dim_het"]))
            print('rotation learning phase')
        trans_abs_val = 0
        total_loss = torch.zeros(1).cuda(params_learning_setup["device"])
        h = 0
        if params_learning_setup["freeze_encoder"] is not None and ep == params_learning_setup["freeze_encoder"]:
            print('freezeing')
            end_to_end_net.freeze_encoder()
        bs = params_learning_setup["bs"]
        for v, d in enumerate(dataloader):
            h+=1
            if phases[ep]:
                # image to image learning
                _, view, _, rot_mat, _, t_vec, _, _ = d
                t_vec = t_vec.cuda(device)
                view = view.cuda(device)
                rot_mat = rot_mat.cuda(device)
                view_augmented = end_to_end_net.augment_batch(view, params_learning_setup["sym_loss"]) # à checker
                end_to_end_net.unfreeze_decoder()
                #end_to_end_net.freeze_encoder()
                model_output_convolved, model_output, trans, est_rot_mat, _, est_het, siren_learned, additional_regul = end_to_end_net.forward(
                    view_augmented, t_vec, rot_mat, rot_mat_axis,pass_het=(ep >= params_learning_setup["start_ep_learn_het"]),
                    known_rot=params_learning_setup["known_rot"], known_trans=params_learning_setup["known_trans"])
                saved_het.append(est_het.detach())
                if params_learning_setup["sym_loss"]:
                    loss = sym_loss(model_output_convolved, view)*bs
                else:
                    if params_learning_setup["loss_type"] == 'l2':
                        if not params_learning_setup["impose_cylinder"]:
                            loss = ((model_output_convolved - view) ** 2).mean()*bs
                        else:
                            loss = (((model_output_convolved[:,0,:,:,:] - view[:,0,:,:,:]) ** 2).mean()*bs
                                    + params_learning_setup["coeff_channel_impose_cyl"] * ((model_output_convolved[:,1,:,:,:] - view[:,1,:,:,:]) ** 2).mean()*bs)
                        """
                        print('view shape', view.shape)
                        print('modf', model_output_convolved.shape)
                        save_4d_for_chimera(to_numpy(model_output_convolved[0]), f'{save_fold}/modout_channel_1.tif')
                        save_4d_for_chimera(to_numpy(model_output_convolved[1]), f'{save_fold}/modout_channel_2.tif')
                        1/0
                        """
                    else:
                        loss = torch.abs(model_output_convolved - view.squeeze()).mean() * bs
                loss += params_learning_setup["regul_volume_param"] * torch.sum(torch.relu(-model_output))
                loss += additional_regul
                if params_learning_setup["heterogeneity"]: # and not params_learning_setup["impose_cylinder"]:
                    loss += params_learning_setup["regul_heterogeneity_param"]* torch.mean(est_het ** 2)
                if not params_learning_setup["known_trans"]:
                    trans_abs_val += np.mean(np.abs(to_numpy(trans)))
                if not params_learning_setup["known_trans"] and params_learning_setup["use_reg_trans"]:
                    coeff_reg_trans /= params_learning_setup["coeff_reg_trans_ratio"]
                    loss += coeff_reg_trans * torch.sum(trans**2)

                est_rot_mats.append(to_numpy(est_rot_mat.squeeze()))
                true_rot_mats.append(to_numpy(rot_mat))

            else:
                # pose to pose learning
                random_rot_mat = []
                bs = saved_het[v].shape[0]
                # generates random rotation matrices
                for b in range(bs):
                    rv = [np.random.randint(360), np.random.randint(180), np.random.randint(360)]
                    rm = get_3d_rotation_matrix(rv)
                    random_rot_mat.append(rm)
                random_rot_mat = torch.FloatTensor(random_rot_mat).cuda(device)
                # generates random translation vector
                if params_learning_setup["known_trans"]:
                    random_trans_vec = 0.00*np.random.randn(bs, 3)
                else:
                    random_trans_vec = 0.09*np.random.randn(bs, 3)
                random_trans_vec = torch.FloatTensor(random_trans_vec).cuda(device)
                end_to_end_net.freeze_decoder()
                #end_to_end_net.unfreeze_encoder()
                est_rot_mat, est_trans = end_to_end_net.forward_rotation(random_rot_mat,random_trans_vec,
                                                                         saved_het[v], params_learning_setup) # forward rotation in decoder and next encoder (reversed order),
                if 1:
                    # outputs an estimated rotation.
                    distance_matrix = torch.mean(torch.linalg.matrix_norm(est_rot_mat - random_rot_mat, dim=(1,2))) #compares it to the input rotation
                    distance_transvec = torch.mean(torch.abs(est_trans - random_trans_vec))
                    loss = 1 / 9 * distance_matrix + 1 / 3 * distance_transvec
                    #total_loss += loss
                    
                else:
                    gen_view, _ = end_to_end_net.forward_decoder(random_rot_mat, random_trans_vec, saved_het[v])
                    gen_view_conv = end_to_end_net.psf_layer.forward(gen_view)

                    est_view, _ = end_to_end_net.forward_decoder(est_rot_mat, est_trans, saved_het[v])
                    est_view_conv = end_to_end_net.psf_layer.forward(est_view)
                    
                    loss = ((est_view_conv - gen_view_conv) ** 2).mean()*bs
                    #loss += params_learning_setup["regul_volume_param"] * torch.sum(torch.relu(-est_view))

                est_rot_mats.append(to_numpy(est_rot_mat.squeeze()))
                true_rot_mats.append(to_numpy(random_rot_mat))

            total_loss += loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            """
            if params_learning_setup["impose_cylinder"]:
                end_to_end_net.cylinder_decoder.uptate_pos_param()
            """

        rpe = compute_relative_pose_error(est_rot_mats, true_rot_mats, params_data_gen["nb_views"])
        rpes.append(rpe)

        # print('pos', end_to_end_net.cylinder_decoder.get_pos_from_pos_param())
        avg_loss = to_numpy(total_loss/params_data_gen["nb_views"])
        if phases[ep]:
            print(f"Step {ep}, mean loss {avg_loss}")
        else:
            print(f"Step {ep}, mean loss pose to pose {avg_loss}")
        print("trans abs val", trans_abs_val/h)
        trans_abs_vals.append(trans_abs_val/h)
        losses.append(avg_loss)
        if phases[ep]:
            losses_i2i.append(avg_loss)
            losses_p2p.append(losses_p2p[-1])
        else:
            losses_i2i.append(losses_i2i[-1])
            losses_p2p.append(avg_loss)
        if ep == 0:
            losses_i2i.pop(0)
            losses_p2p.pop(0)
        x = params_learning_setup["x"]
        if ep%x == x-1:
            inter_losses_save_fold = f'{save_fold}/losses'
            make_dir(inter_losses_save_fold)

            plt.plot(losses)
            plt.savefig(f'{inter_losses_save_fold}/losses_{ep}.png')
            plt.close()

            plt.plot(losses_i2i)
            plt.savefig(f'{inter_losses_save_fold}/losses_i2i_{ep}.png')
            plt.close()

            plt.plot(losses_p2p)
            plt.savefig(f'{inter_losses_save_fold}/losses_p2p_{ep}.png')
            plt.close()
            # print('save to ', save_fold)
            if rot_mat_axis is not None:
                for g in range(1,9):
                    out_i = model_output[0][0][g]
                    out_numpy_i = to_numpy(out_i.view(*([params_data_gen["size"]]*params_data_gen["nb_dim"])))
                    save(f'{save_fold}/recons_ep_{ep}_{g}.tif', out_numpy_i)

            inter_save_fold = f'{save_fold}/ep_{ep}'
            make_dir(inter_save_fold)

            plt.plot(rpes)
            plt.savefig(f'{inter_save_fold}/rpes_{ep}.png')
            plt.close()

            end_to_end_net.curent_inference(data_set, inter_save_fold, params_data_gen, params_learning_setup)
    # est_heterogeneities, est_rot_mats, est_transs, file_names, est_vols, views = end_to_end_net.curent_inference(data_set, params_learning_setup.device, params_data_gen.convention, save_fold)
    plt.plot(losses)
    plt.savefig(f'{save_fold}/losses.png')
    plt.close()

    plt.plot(losses_i2i)
    plt.savefig(f'{save_fold}/losses_i2i.png')
    plt.close()

    plt.plot(losses_p2p)
    plt.savefig(f'{save_fold}/losses_p2p.png')
    plt.close()

    write_array_csv(np.array(losses), f'{save_fold}/losses.csv')
    write_array_csv(np.array(trans_abs_vals), f'{save_fold}/trans_abs_vals.csv')
    end_to_end_net.save_model(save_fold)


def read_views_rot_mats(pth):
    pth_views = f'{pth}/views'
    pth_rot_mats = f'{pth}/rot_mats'
    views = []
    rot_mats = []
    for fn in os.listdir(pth_views):
        view = read_image(f'{pth_views}/{fn}')
        rot_mat = read_csv(f'{pth_rot_mats}/{fn}.csv', first_col=1)
        views.append(view)
        rot_mats.append(rot_mat)
    return views, rot_mats


def read_heterogene_views_synth_data(params_learn_setup, params_data_gen, random_order=False, two_vars=False):
    """le paramètre important à spécifier ici est params_data_gen["pth_views"], qui est le dossier dans lequel sont sauvegardés les
    vues qui vont être lues par la fonction. Il doit contenir autant d'image que params_data_gen["nb_views"]
    Le nom des vues doit être sous la forme suivante : view_{het1}_{rot_vec[0]}_{rot_vec[1]}_{rot_vec[2]}_.tif si le nombre de
    dimensions dans l'espace latent est égal à 1. Si le nombre de dimension de l'espace latent est égal à 2 (two_vars=True),
    le nom des vues doit prendre la forme view_{het1}_{het2}_{rot_vec[0]}_{rot_vec[1]}_{rot_vec[2]}_.tif
    Ces deux formes sont générées respectivement par les fonctions 'heterogene_views_from_centriole' et
    'heterogene_views_from_centriole_2_degrees_of_freedom' dans le fichier python generate_data.py (dossier data_generation)"""
    views = []
    all_file_names = []
    if random_order:
        perm = np.random.permutation(params_data_gen["nb_views"])
    else:
        perm = range(params_data_gen["nb_views"])
    for c in range(params_learn_setup["nb_channels"]):
        views_c, file_names = read_images_in_folder(f'{params_data_gen["pth_views"]}/c{c + 1}', alphabetic_order=False,
                                                    size=params_data_gen["size"])
        file_names = np.array(file_names)[perm]
        views.append(views_c[perm])
        all_file_names.append(file_names)

    for c in range(len(all_file_names) - 1):
        if (all_file_names[c] != all_file_names[c + 1]).any():
            print(f'fn {c}', all_file_names[c])
            print(f'fn {c + 1}', all_file_names[c + 1])
            raise "file names of all channels must be identical"

    file_names = all_file_names[0]
    views = np.array(views)
    views = np.transpose(views, (1, 0, 2, 3, 4))
    dilatation_vals = []
    rot_vecs = []
    rot_mats = []
    nb_views = params_data_gen["nb_views"]
    for v in range(nb_views):
        if not two_vars:
            _, dil_val, rv1, rv2, rv3, _ = file_names[v].split('_')
            dilatation_vals.append(float(dil_val))
        else:
            _, dil_val1, dil_val2, rv1, rv2, rv3, _ = file_names[v].split('_')
            dilatation_vals.append(np.array([float(dil_val1), float(dil_val2)]))
        rot_vec = [float(rv1), float(rv2), float(rv3)]
        rot_vecs.append(rot_vec)
        rot_mat = get_3d_rotation_matrix(rot_vec, convention=params_data_gen["convention"])
        rot_mats.append(rot_mat)
    rot_mats = np.array(rot_mats)
    rot_vecs = np.array(rot_vecs)
    dilatation_vals = np.array(dilatation_vals)
    print('rot vecs', rot_vecs)
    transvecs = np.zeros((nb_views, 3))
    return views, rot_mats, rot_vecs, transvecs, dilatation_vals, file_names

def read_homogene_views_synth_data(params_learn_setup, params_data_gen, random_order=False, channels=False):
    views = []
    all_file_names = []
    if random_order:
        perm = np.random.permutation(params_data_gen["nb_views"])
    else:
        perm = range(params_data_gen["nb_views"])

    if channels:
        for c in range(params_learn_setup["nb_channels"]):
            views_c, file_names = read_images_in_folder(f'{params_data_gen["pth_views"]}/c{c + 1}', alphabetic_order=False,
                                                        size=params_data_gen["size"])
            file_names = np.array(file_names)[perm]
            views.append(views_c[perm])
            all_file_names.append(file_names)

        for c in range(len(all_file_names) - 1):
            if (all_file_names[c] != all_file_names[c + 1]).any():
                print(f'fn {c}', all_file_names[c])
                print(f'fn {c + 1}', all_file_names[c + 1])
                raise "file names of all channels must be identical"
    else:
        views_c, file_names = read_images_in_folder(f'{params_data_gen["pth_views"]}', alphabetic_order=False,
                                                    size=params_data_gen["size"])
        file_names = np.array(file_names)[perm]
        views.append(views_c[perm])
        all_file_names.append(file_names)

    file_names = all_file_names[0]
    views = np.array(views)
    views = np.transpose(views, (1, 0, 2, 3, 4))
    rot_vecs = []
    rot_mats = []
    transvecs = []
    nb_views = params_data_gen["nb_views"]
    for v in range(nb_views):
        _, _, rv1, rv2, rv3, _, tv1, tv2, tv3, _ = file_names[v].split('_')
        rot_vec = [float(rv1), float(rv2), float(rv3)]
        rot_vecs.append(rot_vec)
        rot_mat = get_3d_rotation_matrix(rot_vec, convention=params_data_gen["convention"])
        rot_mats.append(rot_mat)
        trans_vec = [float(tv1), float(tv2), float(tv3)]
        transvecs.append(trans_vec)
    rot_mats = np.array(rot_mats)
    rot_vecs = np.array(rot_vecs)
    transvecs = np.array(transvecs)
    dilatation_vals = np.zeros((nb_views))
    print('rot vecs', rot_vecs)
    print('trans vecs', transvecs)
    return views, rot_mats, rot_vecs, transvecs, dilatation_vals, file_names

#for i,rotations_max in enumerate([[360, 180, 180], [180,180,180], [180,360,360], [360,180,360], [360,360,180],[360,360,360]]):
if __name__ == '__main__':
    from reconstruction_with_dl.test_params.default_params import params_learn_setup, params_data_gen
    from manage_files.paths import PATH_PROJECT_FOLDER, PATH_VIEWS
    from reconstruction_with_dl.test_params.default_params import params_learn_setup, params_data_gen
    params_data_gen["size"] = 50
    params_data_gen["sig_z"] = 3 # standard deviation of psf along z axis
    params_data_gen["sig_xy"] = 1 # standard deviation of psf in xy plane
    params_data_gen["nb_views"] = 250

    params_learn_setup["bs"] = 1 # batch size

    params_learn_setup["nb_epochs_each_phases_ACE_Het"] = [20,20] # can take either a None value or a 2 elements list [n,m].
    # if None, we use only image to image learning phases. If in the form [n,m], alterates between n epochs of image to image
    # learning phases and m epochs of pose to pose learning phases
    params_learn_setup["sym_loss"] = False # use of symetric loss

    params_learn_setup["nb_channels"] = 1
    params_learn_setup['known_rot'] = False #if true learns with known rot
    params_learn_setup["known_trans"] = False #if true learns with known trans

    params_learn_setup["rot_representation"] = "6d" # options : 'axis_angle', 'euler', 'quaternion', '6d' (see top of 'pose_net.py' file)
    params_learn_setup["encoder_name"] = 'holly' #options : 'holly', 'vgg' (see top of pose_net.py file)

    params_learn_setup["impose_cylinder"] = False
    if not params_learn_setup["impose_cylinder"]:
        params_learn_setup["nb_dim_het"] = 1
    else:
        params_learn_setup["nb_dim_het"] = 7
    params_learn_setup["impose_symmetry"] = False
    #params_learn_setup["init"] = f'{PATH_PROJECT_FOLDER}/results_deep_learning/heterogene_views/1_missing_triplets_no_het/c1/gt/4d_vol_channel.tiff'
    for hete in [True]:
        params_learn_setup["heterogeneity"] = hete
        params_learn_setup["nb_epochs"] = 10000
        params_learn_setup["x"] = 220 # save the results every x epochs
        #params_data_gen["pth_views"] = f'{PATH_PROJECT_FOLDER}/results_deep_learning/heterogene_views/bicanal_donut_views_1'
        #params_data_gen["pth_views"] = f'{PATH_PROJECT_FOLDER}/results_deep_learning/heterogene_views/bicanal_donut_viewszero_pos'
        params_data_gen["pth_views"] = f'{PATH_PROJECT_FOLDER}/results_deep_learning/heterogene_views/2_channels'
        #params_data_gen["pth_views"] = f'{PATH_PROJECT_FOLDER}/results_deep_learning/heterogene_views/NPC/example_other_param_0/views' #pth of input data
        #params_data_gen["pth_views"] = f'{PATH_PROJECT_FOLDER}/results_deep_learning/heterogene_views/multi_dim_het_missing_triplets_no_het'
        # defines path of save fold
        save_fold = f'{PATH_PROJECT_FOLDER}/results_deep_learning/centriole_2_channels/test' # path of saved results
        #save_fold = f'{PATH_PROJECT_FOLDER}/results_deep_learning/centriole/impose_symmetry/nb_missing_triplets_{nb_missing_triplets}'
        if not hete:
            save_fold += 'no_het'
        #save_fold = f'{PATH_PROJECT_FOLDER}/results_deep_learning/centriole/diff_max_lenght'
        if params_learn_setup["nb_epochs_each_phases_ACE_Het"] is not None:
            save_fold += '_pose_to_pose'
        params_learn_setup["save_fold"] = save_fold
        views, rot_mats, rot_vecs, transvecs, dilatation_vals, file_names = read_heterogene_views_synth_data(params_learn_setup, params_data_gen, True, False)
        params_learn_setup[
            "coeff_channel_impose_cyl"] = np.sum(views[:,0,:,:,:])/np.sum(views[:,1,:,:,:])  # relative weight of channel 2 wrt to channel 1 in the loss when we impose the tore shape
        data_set = ViewsRandomlyOrientedSimData(views, rot_mats, rot_vecs, transvecs, dilatation_vals,
                                                params_data_gen["size"], params_data_gen["nb_dim"], file_names)
        print('start train')
        train(data_set, params_data_gen, params_learn_setup, None)
        #train_bicanal_cylinder_rep(data_set, params_data_gen, params_learn_setup)


    

    
    

    

            
    

        






    
    
            
    
    





