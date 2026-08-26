import ast
import numpy as np
import torch
from reconstruction_with_dl.data_set_views import get_mgrid
from reconstruction_with_dl.pose_net import to_numpy
import torch.nn as nn
import random
"""
def cylinder_mask(size_xy, size_z):
    grid = make_grid(size_xy, 2)
    distances_to_center = np.sqrt(grid[:,:,0] ** 2 + grid[:,:,1] ** 2)
    mask = (distances_to_center <= 1)
    mask_expanded = np.expand_dims(mask, axis=0)
    mask_expanded = np.repeat(mask_expanded, size_z, axis=0)
    return mask
"""


class CylinderDecoder(nn.Module):
    def __init__(self, grid, params_learn_setup, size):
        """class that represents a decoder that takes the form of a cylinder (more precislely a donut). It is used to replace the siren
        architecture when we have not data enough to get a good reconstruction. The donut form is then used as a strong a priori.
        The encoder then learns to predict the parameter of the donut.
        This parameter of the donut are
            - when we have one channel (lenght, radius, width)
            - when we have two channels (lenght channel 1, lenght channel 2, radius channel 1, radius channel 2,
                                                width channel 1, width channel 2, pos)
            pos is the position of channel 2 with respect t channel 1. We suppose that channel 2 has the same rotation axis than channel 1.
             So we only estimate the position on this rotation axis, that is why pos has a dimension of 1.

        To initialize the cylinder decoder, you need to specifiy a 3d grid (unflatten version), the dictionnary params_learn_setup and the size.
        Here the important parameter to set in params_learn_setup are the min and max values for lenght, radius, width and pos as well as
        the learning rate of pos parameter lr_pos_param (see file default_params.py in folder test_params for the defaul values)
        """
        super().__init__()
        self.grid = grid
        self.size = size
        self.nb_channels = params_learn_setup["nb_channels"]
        p = params_learn_setup["cylinder_maxmin_parameters"]
        self.min_lenght = p["min_lenght"]
        self.max_lenght = p["max_lenght"]
        self.min_radius = p["min_radius"]
        self.max_radius = p["max_radius"]
        self.min_width = p["min_width"]
        self.max_width = p["max_width"]
        self.min_pos = 0
        self.max_pos = p["max_pos"]
        self.pos_param = torch.zeros(3, requires_grad=True)
        self.lr_pos_param = params_learn_setup["lr_pos_param"]

        """
        fc_commun_l1 = nn.Linear(nb_dim_het, 256)
        fc_commun_l2 = nn.Linear(256, 256)
        fc_commun_l3 = nn.Linear(256, 9)
        self.common_regressor = nn.Sequential(fc_commun_l1, nn.LeakyReLU(), fc_commun_l2, nn.LeakyReLU(), fc_commun_l3)

        
        fc_radiuses_l1 = nn.Linear(256, 256)
        fc_radiuses_l2 = nn.Linear(256, nb_channels)
        self.radiuses_regressor = nn.Sequential(fc_radiuses_l1, nn.LeakyReLU(), fc_radiuses_l2)

        fc_lenghts_l1 = nn.Linear(256, 256)
        fc_lenghts_l2 = nn.Linear(256, nb_channels)
        self.length_regressor = nn.Sequential(fc_lenghts_l1, nn.LeakyReLU(), fc_lenghts_l2)

        fc_width_l1 = nn.Linear(256, 256)
        fc_widths_l2 = nn.Linear(256, nb_channels)
        self.widths_regressor = nn.Sequential(fc_width_l1, nn.LeakyReLU(), fc_widths_l2)

        fc_pos_l1 = nn.Linear(256, 256)
        fc_pos_l2 = nn.Linear(256, 3)
        self.pos_regressor = nn.Sequential(fc_pos_l1, nn.LeakyReLU(), fc_pos_l2)
        
        self.layers_fc = [fc_commun_l1, fc_commun_l2, fc_commun_l3]

        for l in self.layers_fc:
            torch.nn.init.xavier_uniform_(l.weight)
            l.bias.data.fill_(random.random() * 0.001)
        """

    def uptate_pos_param(self):
        gradient = self.pos_param.grad
        self.pos_param = torch.tensor(self.pos_param - self.lr_pos_param * gradient, requires_grad=True)


    def get_cylinder_param_from_het_value_2_channels(self, het_val):
        """the parameters predicted by the encoder are not limited in range. This is a probleme, since we knwon that
        the donut size can not be hogher than image size. This finction applies sigmoid functions to the outptut of encoders
        to ensure that donut parameters are in a given range"""
        het_val_sig = torch.sigmoid(het_val)
        lenght = (self.max_lenght - self.min_lenght) * het_val_sig[:, :2] + self.min_lenght
        radius = (self.max_radius - self.min_radius) * het_val_sig[:, 2:4] + self.min_radius
        width = (self.max_width - self.min_width) * het_val_sig[:, 4:6] + self.min_width
        pos = (self.max_pos - self.min_pos) * het_val_sig[:, 6:] + self.min_pos
        return lenght, radius, width, pos

    def get_cylinder_param_from_het_value(self, het_val):
        """same as function above but for a one channel donut"""
        het_val_sig = torch.sigmoid(het_val)
        lenght = (self.max_lenght - self.min_lenght) * het_val_sig[:, 0].squeeze() + self.min_lenght
        radius = (self.max_radius - self.min_radius) * het_val_sig[:, 1].squeeze() + self.min_radius
        width = (self.max_width - self.min_width) * het_val_sig[:, 2].squeeze() + self.min_width
        return lenght, radius, width

    def get_pos_from_pos_param(self):
        pos = (self.max_pos - self.min_pos) * torch.sigmoid(self.pos_param) + self.min_pos
        return pos

    def forward(self, het_val, grid):
        grid_unflatten = nn.Unflatten(-2, (self.size, self.size, self.size))(grid).squeeze()
        #print('het val', het_val)
        #het_val_sig = 1.7*torch.sigmoid(het_val) + 0.04
        if self.nb_channels not in [1,2]:
            raise ValueError(f"When you impose a cylinder, the number of channels should be 1 or 2, here it is equal to {self.nb_channels}")

        if self.nb_channels == 2:
            if het_val.shape[1] != 7:
                raise ValueError(
                    f"When you impose a cylinder and there are 2 channels, the number of dimension of latent space should be 7"
                    f" here it is equal to : {het_val.shape[1]} (2 first : lenghts of cylinders, 2 following radiuses of cylinder, "
                    f"2 following widths of cylinder, 2 last position of cylinder")
            lenght, radius, width, pos = self.get_cylinder_param_from_het_value_2_channels(het_val)
            # pos = self.pos_regressor(features_common)
            # donut = bicannal_donut_cylinder([0.5, 0.2], lenght, [0.1,0.05], grid_unflatten, torch.zeros(3))
            # pos = self.get_pos_from_pos_param()
            """
            radius = [[0.5,0.15]]
            width = [[0.1,0.04]]
            pos = [[-0.1,-0.2,-0.1738]]
            """
            res = []
            if len(grid_unflatten.shape) == 4:
                grid_unflatten = grid_unflatten.unsqueeze(0)
            for b in range(lenght.shape[0]):
                donut_b = bicannal_donut_cylinder(radius[b], lenght[b], width[b], grid_unflatten[b], pos[b])
                res.append(donut_b.unsqueeze(0))
            
            donut = torch.cat(res, dim=0)
            # donut = bicannal_donut_cylinder(radius.squeeze(), lenght.squeeze(), width.squeeze(), grid_unflatten, pos.squeeze())
        else:
            if het_val.shape[1] != 3:
                raise ValueError(
                    f"When you impose a cylinder and there is 1 channels, the number of dimension of latent space should be 3"
                    f" here it is equal to : {het_val.shape[1]} (lenght, width and radius of cylinder)")
            lenght, radius, width = self.get_cylinder_param_from_het_value(het_val)
            # (nb_channels == 1)
            donut = donut_cylinder(radius, lenght, width, grid_unflatten, torch.zeros(3))
            donut.unsqueeze(0) #unsqueeze for channel
            donut = donut.unsqueeze(0) #unsqueeze for  batch
        return donut, 0


def gauss_function(x, mu, std):
    return torch.exp(-(x-mu)**2/(2*std**2))/(std * 2.50662)  # 2.50662 = torch.sqrt(2*torch.pi)


def bicannal_donut_cylinder(radiuses, lenghts, widths, grid, pos):
    donut1 = donut_cylinder(radiuses[0], lenghts[0], widths[0], grid, torch.zeros(3))
    #donut2 = donut_cylinder(radiuses[1], lenghts[1], widths[1], grid, [0.3])
    donut2 = donut_cylinder(radiuses[1], lenghts[1], widths[1], grid, pos)
    donut1_expanded = donut1.unsqueeze(0)
    donut2_expanded = donut2.unsqueeze(0)
    cat_donut = torch.cat((donut1_expanded, donut2_expanded), dim=0)
    return cat_donut


def donut_cylinder(radius, lenght, width, grid, pos):
    distances_to_center = torch.sqrt(grid[:,:,:,0] ** 2 + grid[:,:,:,2]**2)
    donut = gauss_function(distances_to_center, radius, width)
    abs = torch.abs(grid[:,:,:,1] - pos[0])
    lenght_mask = smooth_rectangle_function(abs, lenght)
    donut_lenght =  donut * lenght_mask
    donut_lenght = donut_lenght / torch.max(donut_lenght)
    return donut_lenght


def penalyze(x, val_min, val_max):
    """returns a tensor that contains non zeros values when the element i smaller than vall_min or greater than val_max"""
    return torch.abs(torch.relu(-x + val_min)) + torch.abs(torch.relu(x - val_max))

def smooth_rectangle_function(x, lenght, omega=50):
    return torch.tanh(omega*(x + lenght/2)) + torch.tanh(-omega*(x-lenght/2))


if __name__ == '__main__':
    from manage_files.read_save_files import save, save_multi_channel, save_4d_for_chimera, make_dir
    from manage_matplotlib.graph_setup import set_up_graph
    from pytorch3d.transforms import axis_angle_to_matrix
    from test_params.default_params import params_data_gen
    from manage_files.paths import PATH_PROJECT_FOLDER, PTH_GT
    from data_generation.generate_data import generate_one_view, generate_rot_vec_trans_vec
    from common_image_processing_methods.rotation_translation import discretize_sphere_uniformly
    from classes_with_parameters import ParametersDataGeneration
    from scipy.signal.windows import kaiser_bessel_derived
    import matplotlib.pyplot as plt
    from torch import kaiser_window

    """
    set_up_graph(MEDIUM_SIZE=50)
    X = torch.linspace(-3,3, 300)
    Y = 0.5*smooth_rectangle_function(X, 2)
    plt.plot(X, Y)
    plt.show()
    """


    size = 45
    nb_views = 250
    grid, _ = get_mgrid(size, dim=3)
    pth_donut_views_bicannal = f'{PATH_PROJECT_FOLDER}/results_deep_learning/heterogene_views/bicanal_donut_viewszero_pos'
    make_dir(f'{pth_donut_views_bicannal}/c1')
    make_dir(f'{pth_donut_views_bicannal}/c2')
    uniform_sphere_disc = discretize_sphere_uniformly(10000, 360)
    gt_4d = [[] for c in range(2)]
    lenght1 = []
    lenght2 = []
    for l in np.linspace(0.05, 1.5, nb_views):
        lenght1.append(l)
        lenght_other_cyl = l if l <= 0.5 else 0.5
        lenght2.append(lenght_other_cyl)
        donut_bicanal = bicannal_donut_cylinder([0.5, 0.15], [l, lenght_other_cyl], [0.1, 0.04], grid, [0.2])
        rot_vec, t = generate_rot_vec_trans_vec(params_data_gen, uniform_sphere_disc)
        for c in range(2):
            filtered, _, _, _ = generate_one_view(donut_bicanal[c], rot_vec, t, 3, params_data_gen)
            save(f'{pth_donut_views_bicannal}/c{c+1}/view_{round(l, 3)}_{rot_vec[0]}_{rot_vec[1]}_{rot_vec[2]}_.tif',
                 filtered)
            gt_4d[c].append(to_numpy(donut_bicanal[c]))

    set_up_graph(MEDIUM_SIZE=30)
    plt.scatter(range(250), lenght1, s=5, label='Longueur canal 1')
    plt.scatter(range(250), lenght2, s=5, label='Longueur canal 2')
    plt.grid()
    plt.xlabel('Index des états de croissance')
    plt.legend()
    plt.show()
    1/0
    for c in range(2):
        save_4d_for_chimera(np.array(gt_4d[c]), f'{pth_donut_views_bicannal}/gt_c{c+1}.tiff')



    pth_donut_views = f'{PATH_PROJECT_FOLDER}/results_deep_learning/heterogene_views/donut_views_2'
    make_dir(f'{pth_donut_views}/c1')
    make_dir(f'{pth_donut_views}/gt')
    params_data_gen["sig_z"] = 5
    gt_4d = []
    for l in np.linspace(0.05,1.5, 250):
       donut = donut_cylinder(0.5, l, 0.1, grid, torch.zeros(3))
       rot_vec, t = generate_rot_vec_trans_vec(params_data_gen, uniform_sphere_disc)
       filtered, _, _, _ = generate_one_view(donut, rot_vec, t, 3, params_data_gen)
       save(f'{pth_donut_views}/c1/view_{round(l, 3)}_{rot_vec[0]}_{rot_vec[1]}_{rot_vec[2]}_.tif',
           filtered)
       gt_4d.append(to_numpy(donut))
    save_4d_for_chimera(gt_4d, f'{pth_donut_views}/gt/gt_4d.tiff')
    1/0

    rot_mat = axis_angle_to_matrix(torch.tensor([[0.65,0.45,0.38]]))
    rot_mat = rot_mat[0]
    print('grid shape', grid.shape)
    rotated_grid =  grid @ rot_mat
    donut_bicanal = bicannal_donut_cylinder(torch.tensor([[1,0.30]]), torch.tensor([[1,0.1]]), torch.tensor([[0.1,0.05]]), grid, torch.tensor([[0.1,0.1,-0.1]]))
    save_multi_channel('cylinder_bicanal.tif', to_numpy(donut_bicanal))

