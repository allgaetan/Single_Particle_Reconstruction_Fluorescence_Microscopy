import numpy as np
import torch
import torch.nn as nn

from stage_gaetan.hps import hierarchical_pose_search
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d

class PoseTable(nn.Module):
    """
    Table de correspondance entre les indices de vues et les poses associées
    (pas d'estimation de translations pour le moment)
    """
    def __init__(self, params_learning_setup, params_data_gen):
        super().__init__()
        n_views = params_data_gen["nb_views"]
        nb_dim = params_data_gen["nb_dim"]
        sigma_trans_ker = params_data_gen["sigma_trans_ker"]
        self.unknown_rot = not params_learning_setup["known_rot"]
        self.unknown_trans = not params_learning_setup["known_trans"]
 
        init_6d = torch.zeros(n_views, 6)
        init_6d[:, 0] = 1.0
        init_6d[:, 4] = 1.0
        self.table_rot = nn.Parameter(init_6d.clone(), requires_grad=self.unknown_rot)
        self.table_trans = nn.Parameter(
            sigma_trans_ker * torch.randn(n_views, nb_dim), requires_grad=self.unknown_trans)

    def initialize(self, index, rot_mat, trans_vec=None):
        self.table_rot.data[index] = matrix_to_rotation_6d(rot_mat.to(self.table_rot.device))
        if trans_vec is not None:
            self.table_trans.data[index] = trans_vec.to(self.table_trans.device)

    def forward_hps(self,
                    end_to_end_net,
                    view,
                    index,
                    het,
                    trans,
                    params_learning_setup,
                    params_data_gen,
                    rot_mat_axis=None,
                    true_rot_mat=None):

        if self.unknown_trans:
            print("unknown trans for hps")
            pass
        else:
            if self.unknown_rot:
                print("unknown rot and known trans for hps")
                est_rot_mat, mean_errors = hierarchical_pose_search(end_to_end_net,
                                                    index, 
                                                    view, 
                                                    trans.detach(), 
                                                    het, 
                                                    params_learning_setup,
                                                    params_data_gen, 
                                                    rot_mat_axis=rot_mat_axis)
                self.initialize(index, est_rot_mat, trans) 
            else:
                print("known rot and known trans for hps")
                mean_errors = torch.zeros(params_learning_setup["hps_max_iter"] + 1)
                self.initialize(index, true_rot_mat, trans)

        est_rot_mat = rotation_6d_to_matrix(self.table_rot[index])
        return est_rot_mat, trans, mean_errors

    def forward_sgd(self, index):
        rot_mat = rotation_6d_to_matrix(self.table_rot[index])
        trans = self.table_trans[index]
        if rot_mat.dim() == 2: rot_mat = rot_mat.unsqueeze(0)
        if trans.dim() == 1: trans = trans.unsqueeze(0)
        return rot_mat, trans
        
class ConfTable(nn.Module):
    """
    Table de correspondance entre les indices de vues et les valeurs d'hétérogénéité associées
    (fonctionne seulement en reconstruction homogène pour le moment)
    """
    def __init__(self, params_learning_setup, params_data_gen):
        super().__init__()
        init = torch.zeros((
                params_data_gen["nb_views"], 
                params_learning_setup["nb_dim_het"]
            )).cuda(params_learning_setup["device"])
        self.table_het = nn.Parameter(init, requires_grad=True)

    def initialize(self, index, het_val):
        self.table_het.data[index] = torch.as_tensor(het_val).float().to(self.table_het.device)

    def forward_sgd(self, index):
        return self.table_het[index]