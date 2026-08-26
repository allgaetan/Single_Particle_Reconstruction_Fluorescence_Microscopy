import torch
import numpy as np

def gaussian_mixture(grid, centers, cov, nb_dim, device):
    gaussian_mixture = torch.zeros(grid.shape[:-1]).cuda(device)
    print('gauss mixture shape', gaussian_mixture.shape)
    for i in range(centers.shape[0]):
        g = nd_gaussian(grid, centers[i], cov, nb_dim)
        gaussian_mixture += g.squeeze()
    return gaussian_mixture


def truncated_gaussian_mixture(grid, coeffs, centers, covs, nb_dim, alpha, return_gaussians=False):
    """computes the values of a multivariate gaussian mixture on a 3d grid. It truncates the gaussians around their centers
        IN :
        - grid : 4d array of shape  (d, h, w, 3) set of 3d voxels organised in a 3d volume. Values are computed on these voxels
        - centers : array of shape (Nb_gaussians, 3,1)
        - covs : covariance matrices of the gaussians, shape (Nb_gaussians, 3,3)
        - coeffs : linear coefficient of each gaussian
        OUT : array of shape (d, h, w) containing the values taken by the gaussian mixture at the voxels on the grid"""
    shp = tuple([grid.shape[d] for d in range(nb_dim)])
    gauss_mixture = np.zeros(shp)
    gaussians = []
    for i in range(coeffs.shape[0]):
        gaussian = truncated_nd_gaussian(grid, gauss_mixture, coeffs[i], centers[i], covs[i],  nb_dim, alpha, return_gaussians)
        if return_gaussians:
            gaussians.append(gaussian)
    return gauss_mixture, gaussians


def truncated_nd_gaussian(grid, gaussian_mixture, coeff, center, cov, nb_dim, alpha, return_gaussian=False):
    """computes the values of a multivariate gaussian on a 3d grid. It truncates the gaussian around the center
    IN :
    - grid : 4d array of shape (d, h, w, 3) set of 3d voxels organised in a 3d volume. Values are computed on these voxels
    - center : array of shape (3)
    - cov : covariance matrix of the gaussian, shape (3,3)
    - half_size_cut : half_size of the cut
    OUT : array of shape (d, h, w) containing the values taken by the gaussian at the voxels on the grid"""
    sz = grid.shape[0]
    grid_step = 2/(sz - 1)
    center_int = (center - 1)/grid_step + sz
    # center_int = [int(grid_reverse_transform(center[d])) for d in range(nb_dim)]
    half_size_cut = torch.round(alpha * torch.sqrt(torch.diagonal(cov))/grid_step)
    mins = torch.maximum(torch.tensor(0), center_int - half_size_cut)
    mins = torch.minimum(mins, torch.tensor(sz))
    maxs = torch.minimum(torch.tensor(sz), center_int + half_size_cut)
    maxs = torch.maximum(torch.tensor(0), maxs)
    for d in range(nb_dim):
        cut_grid = torch.narrow(cut_grid, d, mins[d], maxs[d] - mins[d])
    if nb_dim == 2:
        gaussian_mixture[mins[0]:maxs[0], mins[1]:maxs[1]] += coeff * nd_gaussian(cut_grid, center, cov, nb_dim)
    else:
        gaussian_mixture[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]] += coeff * nd_gaussian(cut_grid, center, cov, nb_dim)
    
    
def nd_gaussian_with_points(points, center, cov):
    centered_grid = points - center
    centered_grid = centered_grid.T
    inv_cov = torch.linalg.inv(cov)
    center_grid_dot_cov = torch.matmul(inv_cov.float(), centered_grid.float())
    A = (center_grid_dot_cov * centered_grid).sum(axis=0)
    gauss = torch.exp(-A / 2)
    return gauss


def flattens_grid(grid, nb_dim):
    shp = grid.shape
    size_flatten = 1
    for i in range(nb_dim):
        size_flatten *= shp[i]
    flatten_grid = grid.reshape((size_flatten, nb_dim))
    return flatten_grid, shp


def nd_gaussian(grid, center, cov, nb_dim):
    flatten_grid = grid.view(-1, nb_dim)
    gauss = nd_gaussian_with_points(flatten_grid, center, cov)
    sz = grid.shape[0]
    reshape_dim = tuple([-1] + [sz] * nb_dim)
    return gauss.reshape(reshape_dim)


def make_grid(size, nb_dim, mi=-1., ma=1.):
    slices = []
    for d in range(nb_dim):
        slices.append(slice(0,size, 1))
    slices = tuple(slices)
    transpose_idx = list(range(1,nb_dim+1))
    transpose_idx.append(0)
    grid = np.mgrid[slices].transpose(*transpose_idx)
    if mi != None and ma != None:
        grid_step = (ma-mi) / (size-1)
        grid = grid_step * (grid-size+1) + ma
    return grid


if __name__ == '__main__':
    from time import time
    from torchvision.ops import MLP
    small_grid_prop = 5
    size = 50
    small_grid_size = size//small_grid_prop
    grid_step = 1/(size-1)
    nb_dim = 3
    small_grid = torch.tensor(make_grid(small_grid_size, nb_dim, mi=-0.5/small_grid_prop, ma=0.5/small_grid_prop)).cuda()

    sigma_z = 5
    cov = torch.eye(nb_dim).cuda()
    cov[0,0] = sigma_z**2
    cov *= grid_step**2

    one_gaussian = nd_gaussian(small_grid, torch.zeros(nb_dim).cuda(), cov, nb_dim).squeeze()

    grid = torch.tensor(make_grid(size, nb_dim, mi=-0.5, ma=0.5)).cuda()
    nb_centers = 5

    mlp = MLP(1, [10, nb_centers * 3]).cuda()
    x = torch.tensor(0).cuda().float()
    optimizer = torch.optim.Adam([{'params': mlp.parameters()}])
    im_ref = torch.rand([size]*nb_dim).cuda()
    for it in range(50):
        centers = mlp(x.unsqueeze(0)).view(1, nb_centers, nb_dim).squeeze()
        centers = torch.clip(centers, min=-0.4, max=0.4
                             )
        print('centers', centers)
        gmm = torch.zeros([size] * nb_dim).cuda()
        for i in range(len(centers)):
            print('i', i)
            t = time()
            c_pixel = (centers[i] / grid_step + size //2).type(torch.int64)
            mins = c_pixel - small_grid_size//2
            maxs = c_pixel + (small_grid_size - small_grid_size//2)
            gmm[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]] += one_gaussian
            print('temps', time()-t)
        print('centers grad', centers.requires_grad)
        print('gmm grad', gmm.requires_grad)
        loss = torch.sum((gmm - im_ref)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


