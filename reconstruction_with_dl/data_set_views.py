import matplotlib.pyplot as plt
import torch
import numpy as np
from common_image_processing_methods.others import resize
from torch.utils.data import Dataset
from data_generation.generate_data import generate_data, generate_one_view
from common_image_processing_methods.rotation_translation import get_rotation_matrix
from manage_files.read_save_files import save


def from_numpy_float32(arr):
    return torch.from_numpy(arr.astype(np.float32))


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid_flatten = mgrid.reshape(-1, dim)
    return mgrid, mgrid_flatten


from common_image_processing_methods.rotation_translation import rotation


def duplicate_view_by_rotating(gt, rot_vec, trans_vec, params_data_gen, symmetry_c=9):
    rotated_views = []
    # symmetry_rot_mats = [get_rotation_matrix([0, 360 * k / 9, 0], params_data_gen.convention) for k in range(9)]
    for c in range(symmetry_c):
        rot_vec_sym = [rot_vec[0], rot_vec[1], rot_vec[2]+360 * c / symmetry_c]
        # rot_mat = torch.FloatTensor(rot_mat).cuda(0)
        # rot_mat = get_rotation_matrix([rot_vec[0], rot_vec[1], rot_vec[2] + 360 * c / 9], 'ZXZ')
        # rot_mat = torch.FloatTensor(rot_mat).cuda(0)
        # rotated_view, _ = rotation(gt, rot_mat)
        rotated_view, _ = generate_one_view(gt, rot_vec_sym, trans_vec, 3, params_data_gen)
        rotated_views.append(rotated_view)
        # save(f'view_rot_{c}.tif', rotated_view)
    rotated_views = np.array(rotated_views)
    return rotated_views


class ViewsRandomlyOriented(Dataset):
    def __init__(self, views, size, nb_dim, file_names):
        """represents a data set of real data views. To initialize this class, you need to specify the input views as a
        4d array (nb_views, nb_channels, size, size, size) and the list of corresponding file_names"""
        super().__init__()
        nb_views = len(views)
        self.nb_views = nb_views
        # self.nb_channels = views.shape[1]
        self.views = from_numpy_float32(views.real)
        #self.ht_transforms = torch.zeros(tuple([nb_views, 1] + nb_dim * [size]))
        self.file_names = file_names
        """
        self.views = torch.zeros(tuple([nb_views, 1] + nb_dim * [size]))
        for v in range(len(views)):
            self.views[v] = from_numpy_float32(views[v].real).unsqueeze(0)
            self.ht_transforms[v] = dht(from_numpy_float32(views[v].real)).unsqueeze(0)
        """

    def __len__(self):
        return len(self.views)

    def get_two_cons_views(self, idx):
        if idx < len(self.views)-1:
            two_cons_views = self.views[idx:idx + 2]
        else:
            two_cons_views = torch.cat([self.views[-1].unsqueeze(0), self.views[0].unsqueeze(0)], dim=0)
        return two_cons_views

    def __getitem__(self, idx):
        return (idx, self.views[idx], self.get_two_cons_views(idx), torch.zeros(1),
                torch.zeros(1), torch.zeros(3), torch.zeros(1), self.file_names[idx])


class ViewsRandomlyOrientedSimData(ViewsRandomlyOriented):
    def __init__(self, views, rot_mats, rot_vecs, transvecs, dilatation_vals, size, nb_dim, file_names):
        """shape views : (nb_views, nb_channels, s, s, s)
        shape rot_mats : (nb_views, 3,3)  (true rot_mat associated to the views, used when the algorithms suppose that pose are known)
        shape rot_vecs : (nb_views, 3)
        shape trans_vecs : (nb_views, 3)
        dilatation_vals : true heterogeneity parameters associated to each views. Used in the function curent_inference of class
        End_to_end_architecture_volume to compare the estimated heterogeneity parameter to the true one. Shape (nb_views, h), with h
        the size of latent space
        size : size of images
        nb_dim : number of dimension (=3 or eventually 2)
        file_names : file_names (not path) of views
        """
        super().__init__(views, size, nb_dim, file_names)
        self.rot_vecs = from_numpy_float32(rot_vecs * np.pi / 180)
        grid_step = 2 / size
        self.trans_vecs = from_numpy_float32(transvecs) * grid_step
        self.rot_mats = from_numpy_float32(np.array(rot_mats))
        self.dilatation_vals = dilatation_vals
        _, reference_grid = get_mgrid(size, nb_dim)
        self.rotated_grids = []
        for v in range(len(views)):
            rot_mat_tensor = self.rot_mats[v]
            t_torch = self.trans_vecs[v]
            rotated_grid = torch.matmul(reference_grid, rot_mat_tensor) - torch.matmul(t_torch, rot_mat_tensor)
            self.rotated_grids.append(rotated_grid)

    def __getitem__(self, idx):
        return idx, self.views[idx], self.get_two_cons_views(idx), self.rot_mats[idx], self.rot_vecs[idx], self.trans_vecs[idx], \
            self.dilatation_vals[idx], self.file_names[idx]


if __name__ == '__main__':
    from manage_files.read_save_files import read_image
    #from manage_matplotlib.plot_orthofonal_views import plot_ortho
    from classes_with_parameters import ParametersDataGeneration
    from manage_files.read_save_files import save
    pth = '/home/eloy/Documents/stage_reconstruction_spfluo/results_deep_learning/heterogeneity_centriole/views/' \
          'small_heterogeneity_300/gt_dilated/view_404.8_.tif'

    gt = read_image(pth)
    views, rot_vecs, _, _, _ = generate_data(gt, ParametersDataGeneration(convention='ZXZ'))

    view = views[0]
    rot_vec = rot_vecs[1]
    print('rot vace', rot_vec)
    rotated_views = []
    for c in range(9):
        rot_mat = get_rotation_matrix([rot_vec[0], rot_vec[1], rot_vec[2] + 360 * c / 9], 'ZXZ')
        # rot_mat = torch.FloatTensor(rot_mat).cuda(0)
        rotated_view, _ = rotation(gt, rot_mat)
        rotated_views.append(rotated_view)
        save(f'im_rotated_{c}.tif', rotated_view)

    rotated_views = np.array(rotated_views)




