import numpy as np
try:
    import cupy as cp
    from cupyx.scipy.ndimage import affine_transform as affine_transform_gpu
except:
    pass
from scipy.ndimage.interpolation import affine_transform

from scipy.spatial.transform import Rotation as R
import numbers

interp_order = 3


def angular_diff(target, source):
    a = target - source
    a = np.mod(a + np.pi, 2*np.pi) - np.pi
    return a


def get_rot_vec_from_3d_rot_mat(rot_mat, convention):
    r = R.from_matrix(rot_mat)
    rot_vec = r.as_euler(convention, degrees=True)
    return rot_vec


def get_angle_from_2d_rot_mat(rot_mat):
    cos = rot_mat[0,0]
    sin = rot_mat[1,0]
    angle = np.arccos(cos)
    angle_degree = 180*angle/np.pi
    if sin >= 0:
        return angle_degree
    else:
        return -angle_degree


def get_rot_vec_from_rot_mat(rot_mat, convention):
    if rot_mat.shape == (2,2):
        return get_angle_from_2d_rot_mat(rot_mat)
    elif rot_mat.shape == (3,3):
        return get_rot_vec_from_3d_rot_mat(rot_mat, convention)
    else:
        print("shape of rot mat must be either (2,2) or (3,3)")
        print("rot mat :", rot_mat)
        raise ValueError


def get_2d_rotation_matrix(angle):
    angle = np.pi * angle /180
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
    return rot_mat


def point_cloud_rotation(point_cloud, rot_mat):
    return (rot_mat @ point_cloud.T).T


def get_rotation_matrix(rot_vec, convention='zxz'):
    if hasattr(rot_vec, '__iter__') and len(rot_vec) == 3:
        return get_3d_rotation_matrix(rot_vec, convention)

    else:
        return get_2d_rotation_matrix(rot_vec)


def get_3d_rotation_matrix(rot_vec, convention='zxz'):
    """transforms the rotation vector into a rotation matrix"""
    if convention == 'mrp':
        r = R.from_mrp(rot_vec)
    else:
        r = R.from_euler(convention, rot_vec, degrees=True)
    rot_mat = r.as_matrix()
    return rot_mat


def translation(image, trans_vec, order=interp_order):
    nb_dim = len(image.shape)
    trans_vec = np.array(trans_vec)
    return affine_transform(image, np.eye(nb_dim), -trans_vec, order=order)


def rotation(volume, rot_mat, order=interp_order, trans_vec=None):
    """apply a rotation around center of image"""
    if trans_vec is None:
        trans_vec = np.zeros(len(volume.shape))
    c = np.array([size//2 for size in volume.shape])
    rotated = affine_transform(volume, rot_mat.T, c-rot_mat.T @ (c + trans_vec), order=order, mode='nearest')
    return rotated, rot_mat


def rotation_gpu(volume, rot_mat, order=interp_order, trans_vec=None):
    """apply a rotation around center of image"""
    if trans_vec is None:
        trans_vec = cp.zeros(len(volume.shape))
    c = cp.array([size//2 for size in volume.shape])
    rotated = affine_transform_gpu(volume, rot_mat.T, c-rot_mat.T @ (c + trans_vec), order=order, mode='nearest')
    return rotated, rot_mat


def discretize_sphere_uniformly(nb_view_dir, nb_angles=20):
    ''' Generates a list of the two first euler angles that describe a uniform discretization of the sphere with the Fibonnaci sphere algorithm
    :param N: number of points
    '''

    goldenRatio = (1 + 5 ** 0.5) / 2
    i = np.arange(0, nb_view_dir)
    theta = np.mod(2 * np.pi * i / goldenRatio,2*np.pi)
    phi = np.arccos((1 - 2 * (i + 0.5) / nb_view_dir))
    psi = np.linspace(0,2*np.pi, nb_angles)
    """
    x, y, z = conversion_2_first_eulers_angles_cartesian(theta, phi, False)
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)  # plot the point (2,3,4) on the figure
    plt.show()
    """
    return theta*180/np.pi,phi*180/np.pi, psi*180/np.pi


def conversion_2_first_eulers_angles_cartesian(theta, phi, degrees=True):
    if degrees:
        theta = theta*np.pi/180
        phi = phi*np.pi/180
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    return x,y,z


def conversion_polar_cartesian(rho, theta, phi, degrees=False):
    if degrees:
        theta = theta*np.pi/180
        phi = phi*np.pi/180
    x, y, z = rho*np.cos(theta)*np.sin(phi), rho*np.sin(theta) * np.sin(phi), rho*np.cos(phi)
    return x,y,z


def conversion_cartesian_polar(x,y,z, degree=False):
    rho = np.sqrt(x**2+y**2+z**2)
    phi = np.arccos(z/rho)
    if x > 0:
        theta = np.arctan(y/x)
    elif x < 0 and y >=0:
        theta = np.arctan(y/x) + np.pi
    elif x < 0 and y < 0:
        theta = np.arctan(y/x) - np.pi
    elif x == 0 and y > 0:
        theta = np.pi/2
    elif x == 0 and y < 0:
        theta = -np.pi/2
    else:
        theta = None
    if degree:
        if theta!=None:
            theta = 180*theta/np.pi
        phi = 180*phi/np.pi
    return rho, theta, phi


if __name__ == '__main__':
    from manage_files.read_save_files import read_image, save
    from manage_files.paths import PATH_RESULTS_SUMMARY, PATH_PROJECT_FOLDER

    im = read_image(
        '/home/eloy/Documents/stage_reconstruction_spfluo/results_deep_learning/centriole/missing_triplet_impose_sym'
        '/init_vol.tif')
    for c in range(9):
        rot_mat_sym = get_rotation_matrix([0, 0, 360 * c / 9], 'XYZ')
        rot_im, _ = rotation(im, rot_mat_sym)
        save(f'{PATH_PROJECT_FOLDER}/rot_im_{c}.tif', rot_im)
    1/0


    th_root = "/home/eloy/Documents/documents latex/these/images/homogene_recons_fluo_fire"
    for gt_name in ["recepteurs_AMPA"]:
        for nm in ["recons", "recons_pose_to_pose", "recons_sym_loss"]:
            fn = f'{pth_root}/{gt_name}/{nm}_registered_gradient.tif'
            im = read_image(fn)
            rot_mat = get_rotation_matrix([90, 0, 0])
            im_rotated, _ = rotation(im, rot_mat)
            save(f'{pth_root}/{gt_name}/{nm}_rotated.tif', im_rotated)


    1/0



    1/0
    from manage_files.paths import *
    vol = np.random.random((50,50,50))
    rot_mat = get_3d_rotation_matrix([90,0,0])
    from time import time
    t= time()
    rotation(vol, rot_mat)
    print('temps', time()-t)
    import torch
    from manage_files.read_save_files import read_image, save
    from common_image_processing_methods.others import resize



    1/0

    # pth_root = f'{PATH_RESULTS_HPC}/test_gt'
    pth_root = '/home/eloy/Documents/stage_reconstruction_spfluo/ground_truths'
    for gt_name in gt_names:
        pth_im = f'{pth_root}/{gt_name}.tif'
        # pth_im = f'{pth}/recons_registered.tif'
        im = read_image(pth_im)
        rot_mat = get_3d_rotation_matrix([0, 0, 180], convention='XYZ')
        rot_mat_2 = get_3d_rotation_matrix([180, 180, 0], convention='XYZ')
        im_rotated, _ = rotation(im, rot_mat)
        im_rotated_2, _ = rotation(im, rot_mat_2)
        im_tens = torch.from_numpy(im)
        im_tens_flipped = torch.flip(torch.flip(im_tens, [0,1]), [1,2])
        im_tens_flipped_2 = torch.flip(torch.flip(torch.flip(im_tens, [1, 2]), [0, 1]), [0,2])
        im_tens_flipped_3 = torch.flip(im_tens, [0, 2])
        im_tens_flipped_4 = torch.flip(im_tens, [0, 1])
        im_tens_flipped_5 = torch.flip(im_tens, [1, 2])

        save(f'{pth_root}/{gt_name}_flipped.tif', im_tens_flipped.numpy())
        save(f'{pth_root}/{gt_name}_flipped_2.tif', im_tens_flipped_2.numpy())
        save(f'{pth_root}/{gt_name}_flipped_3.tif', im_tens_flipped_3.numpy())
        save(f'{pth_root}/{gt_name}_flipped_4.tif', im_tens_flipped_4.numpy())
        save(f'{pth_root}/{gt_name}_flipped_5.tif', im_tens_flipped_5.numpy())
        save(f'{pth_root}/{gt_name}_rotated.tif', im_rotated)
        save(f'{pth_root}/{gt_name}_rotated_2.tif', im_rotated_2)

    1 / 0



    1/0


    print(conversion_polar_cartesian(1, 0, np.pi/2))


    rho, theta, phi = conversion_cartesian_polar(1,0,0)
    print(rho)
    print(theta)
    print(phi)




    X = np.array([1,7])
    Y = np.array([2,5])
    Z = np.array([1,4])
    conversion_cartesian_polar(X, Y, Z)


    from manage_matplotlib.colors import gen_colors
    rho, theta, phi = 2, 0.5, 0.3
    x, y, z = conversion_polar_cartesian(rho, theta, phi)
    rho_est, theta_est, phi_est = conversion_cartesian_polar(x, y, z)
    print('rho est', rho_est)
    print('theta est', theta_est)
    print('phi est', phi_est)


