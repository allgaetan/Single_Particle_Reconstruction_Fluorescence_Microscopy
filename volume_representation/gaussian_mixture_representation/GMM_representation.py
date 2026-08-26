import numpy.random as rd
import numpy as np
from common_image_processing_methods.rotation_translation import  point_cloud_rotation
from volume_representation.gaussian_mixture_representation.GMM_grid_evaluation import gaussian_mixture_with_points, gaussian_mixture
from skimage.registration import phase_cross_correlation
from common_image_processing_methods.barycenter import compute_barycenter
from volume_representation.gaussian_mixture_representation.GMM_grid_evaluation import make_grid
from manage_files.read_save_files import save
from common_image_processing_methods.registration import registration_exhaustive_search
from manage_files.read_save_files import write_array_csv
from manage_files.read_save_files import read_image
# from volume_representation.gaussian_mixture_representation.GMM_image_fitting import Gmm_rep_of_view


class GMM_representation():
    def __init__(self, nb_gaussians, sigma, nb_dim, size, cov_PSF, threshold_gaussians):
        centers, coeffs = random_init(nb_gaussians, nb_dim)
        self.nb_gaussians = nb_gaussians
        self.centers = centers
        self.coeffs = coeffs
        self.grid = make_grid(size, nb_dim)
        self.size = size
        self.grid_step = 2/(size-1)
        self.nb_dim = nb_dim
        self.sigma = sigma # *self.grid_step
        self.cov_PSF = cov_PSF
        self.alpha = 3
        self.threshold_gaussians = threshold_gaussians
        self.cov = self.sigma**2*np.eye(self.nb_dim)

    def get_energy(self, rot_mat, trans_vec, view, recorded_shifts, view_idx, save_shift=False, known_trans=True, compute_gaussians=False,
                   interp_order=None):
        rotated_centers = point_cloud_rotation(self.centers, rot_mat)
        rotated_translated_centers = rotated_centers
        if known_trans:
            rotated_translated_centers += self.grid_step*trans_vec
        covs = [self.cov + self.cov_PSF for _ in range(len(self.centers))]
        ratio = (np.linalg.det(self.cov) / np.linalg.det(self.cov + self.cov_PSF)) ** 0.5
        mixture, gaussians = gaussian_mixture(self.grid, self.coeffs, rotated_translated_centers, covs, self.nb_dim,
                                              self.alpha, compute_gaussians)
        save('mixture.tif', mixture)
        mixture*=ratio
        if not known_trans:
            shift, _, _ = phase_cross_correlation(view, mixture)
            rotated_translated_centers += self.grid_step*shift
            mixture, gaussians = gaussian_mixture(self.grid, self.coeffs, rotated_translated_centers, covs, self.nb_dim,
                                                  self.alpha, compute_gaussians)
            mixture *= ratio
            if save_shift:
                recorded_shifts[view_idx].append(shift)

        diff = mixture - view
        energy = np.sum(diff**2)
        variables_used_to_compute_gradient = [diff, gaussians, rotated_translated_centers, covs]
        return energy, variables_used_to_compute_gradient

    def regularization(self, nb_closest):
        reg = 0
        gradient_reg = np.zeros(self.centers.shape)
        dist_mat, closest_nghbrs = find_closest_neighbors(self.centers, nb_closest)
        for i in range(len(self.centers)):
            diff_tot = 0
            for j in range(nb_closest):
                diff = self.centers[i, :] - self.centers[closest_nghbrs[i][j], :]
                diff_tot += diff
                reg += np.sum(diff ** 2)
            gradient_reg[i, :] = 2 * diff_tot
        return gradient_reg / len(self.centers), reg / len(self.centers)

    def one_gd_step(self, rot_mat, trans_vec, views, lr, known_trans, view_idx, recorded_shifts, reg_coeff=0, interp_order=None):
        view = views[view_idx]
        energy, variables_used_to_compute_gradient = \
            self.get_energy(rot_mat, trans_vec, view,recorded_shifts, view_idx,save_shift=True,
                            known_trans=known_trans, compute_gaussians=True)
        diff, gaussians, rotated_translated_centers, covs = variables_used_to_compute_gradient
        grads_coeff = np.zeros(len(self.coeffs))
        grads_center = np.zeros((len(self.coeffs), self.nb_dim))
        for i in range(len(self.coeffs)):
            centered_grid = self.grid - rotated_translated_centers[i]
            gauss_diff = gaussians[i] * diff
            grad_coeff = 2 * np.sum(gauss_diff)
            grads_coeff[i] = grad_coeff
            X = np.multiply(centered_grid, np.expand_dims(gauss_diff, self.nb_dim))
            summ = X.sum(axis=tuple(range(self.nb_dim)))
            grad_center = self.coeffs[i] * rot_mat.T @ np.linalg.inv(covs[i]) @ summ
            grads_center[i] = grad_center
        if reg_coeff != 0:
            grad_reg, energy_reg = self.regularization(max(len(self.centers) // 5, 1))
        else:
            energy_reg = 0
            grad_reg = np.zeros(self.centers.shape)
        """
        for j in range(len(grads_coeff)):
            print('j')
            print(grads_coeff)
        """
        self.coeffs = self.coeffs - lr*grads_coeff
        self.centers = self.centers - lr*grads_center #-10**-3*reg_coeff*grad_reg
        self.coeffs[self.coeffs<0] = 0
        if view_idx == len(views)-1:
            self.suppress_unuseful_gaussians()
        if not known_trans and (view_idx == len(views)-1):
            self.center_barycenter()
        return energy+reg_coeff*energy_reg

    def evaluate_on_grid(self):
        covs = [self.cov for _ in range(len(self.coeffs))]
        gm, _ = gaussian_mixture(self.grid, self.coeffs, self.centers, covs, self.nb_dim, self.alpha)
        return gm

    def center_barycenter(self):
        barycenter = compute_barycenter(self.evaluate_on_grid())
        image_center = np.array([self.size // 2 for _ in range(self.nb_dim)])
        trans_vec = image_center - barycenter
        self.centers = self.centers + self.grid_step * trans_vec

    def suppress_unuseful_gaussians(self):
        new_coeffs = []
        new_centers = []
        for i in range(len(self.coeffs)):
            is_unuseful = False
            if self.coeffs[i] < self.threshold_gaussians:
                is_unuseful = True
            else:
                for d in range(self.nb_dim):
                    if self.centers[i][d] > 1 or self.centers[i][d] < -1:
                        is_unuseful = True
            if not is_unuseful:
                new_coeffs.append(self.coeffs[i])
                new_centers.append(self.centers[i])
        # self.nb_gaussian = len(new_centers)
        if len(new_centers) == 0:
            print('no gaussians left')
            new_coeffs, new_centers = np.array([0]), np.zeros((1,self.nb_dim))
        self.centers = np.array(new_centers)
        self.coeffs = np.array(new_coeffs)

    def save_gmm_parameters(self, fold_path):
        write_array_csv(self.centers, f'{fold_path}/centers.csv')
        write_array_csv(self.coeffs, f'{fold_path}/coeffs.csv')

    def split_gaussians(self, nb_gaussians_ratio, sigma_ratio):
        self.sigma /= sigma_ratio
        nb_gaussians = int(len(self.coeffs) * nb_gaussians_ratio)
        new_coeffs = []
        new_centers = []
        for i in range(len(self.coeffs)):
            center_array = np.array(self.centers[i])
            for _ in range(int(nb_gaussians_ratio)):
                new_coeffs.append(self.coeffs[i] / nb_gaussians_ratio)
                new_c = np.random.multivariate_normal(center_array, 0.01 * np.eye(self.nb_dim))
                new_centers.append(new_c)
        while len(new_centers) < nb_gaussians:
            rd_idx = np.random.randint(0, len(self.centers))
            rd_c = self.centers[rd_idx]
            new_c = np.random.multivariate_normal(rd_c, 0.01 * np.eye(self.nb_dim))
            new_centers.append(new_c)
            new_coeffs.append(self.coeffs[rd_idx] / nb_gaussians_ratio)
        self.coeffs = np.array(new_coeffs)
        self.centers = np.array(new_centers)

    def register_and_save(self, output_dir, output_name, ground_truth_path=None):
        """
        for i in range(len(self.centers)):
            print('centers')
            print(i)
            print(self.centers[i])
        """
        gm_evaluated = self.evaluate_on_grid()
        path = f'{output_dir}/{output_name}.tif'
        save(path, gm_evaluated)
        if ground_truth_path is not None:
            registration_exhaustive_search(ground_truth_path, path, output_dir, f'{output_name}_registered', self.nb_dim)
            registration_exhaustive_search(ground_truth_path, f'{output_dir}/{output_name}_registered.tif', output_dir,
                                           f'{output_name}_registered_gradient', 3,
                                           sample_per_axis=40, gradient_descent=True)
            gm_evaluated = read_image(f'{output_dir}/recons_registered_gradient.tif')
        return gm_evaluated

    def init_with_average_of_views(self, views):
        views = np.array(views)
        avg_view = np.mean(views, axis=0)
        centers = []
        coeffs = []
        GM = np.zeros(avg_view.shape)
        for _ in range(self.nb_gaussians):
            grid = make_grid(self.size, self.nb_dim)
            new_center, coeff = arg_max_array(avg_view - GM)
            new_center = self.grid_step * (new_center - self.size + 1) + 1
            centers.append(new_center)
            coeffs.append(coeff)
            covs = [self.cov for _ in range(len(coeffs))]
            GM, _ = gaussian_mixture(grid, np.array(coeffs), np.array(centers), covs, self.nb_dim, 3)
        self.centers, self.coeffs = np.array(centers), np.array(coeffs)


class GMM_rep_volume_and_views(GMM_representation):
    def __init__(self, nb_gaussians, sigma, nb_dim, size, cov_PSF, threshold_gaussians):
        super().__init__(nb_gaussians, sigma, nb_dim, size, cov_PSF, threshold_gaussians)

    def get_energy(self, rot_mat, trans_vec, view, recorded_shifts, view_idx, save_shift=False, known_trans=True, compute_gaussians=True,
                   interp_order=None):
        rotated_centers = point_cloud_rotation(self.centers, rot_mat)
        rotated_translated_centers = rotated_centers
        if known_trans:
            rotated_translated_centers += self.grid_step * trans_vec
        covs = 2*np.array([self.cov + self.cov_PSF for _ in range(len(self.centers))])
        ratio = (np.linalg.det(self.cov) / np.linalg.det(self.cov + self.cov_PSF)) ** 0.5
        mixture1, gaussians1 = gaussian_mixture_with_points(rotated_translated_centers, self.coeffs, rotated_translated_centers, covs)
        E1 = np.sum(mixture1*self.coeffs)
        mixture2, gaussians2 = gaussian_mixture_with_points(rotated_translated_centers, view.coeffs, view.centers, covs)
        E2 = np.sum(mixture2*self.coeffs)
        energy = ratio ** 2 * E1 - 2 * ratio * E2
        # energy = E1 - 2*E2
        variable_used_to_compute_gradient = [mixture1, mixture2, gaussians1, gaussians2, covs[0], view, rotated_translated_centers, ratio]
        return energy, variable_used_to_compute_gradient

    def one_gd_step(self, rot_mat, trans_vec, views, lr, known_trans, view_idx, recorded_shifts, reg_coeff=0, interp_order=None):
        energy, variable_used_to_compute_gradient = self.get_energy(rot_mat, trans_vec, views[view_idx], recorded_shifts, view_idx,
                                                                    known_trans=known_trans)
        mixture1, mixture2, gaussians1, gaussians2, cov, view, rotated_translated_centers, ratio = variable_used_to_compute_gradient
        grad_coeffs = ratio**2*mixture1 - 2 * ratio * mixture2
        #grad_coeffs = mixture1 - 2*mixture2
        self.coeffs = self.coeffs - lr * grad_coeffs
        centers_diffs1 = two_by_two_differences(self.centers, self.centers)
        centers_diffs2 = two_by_two_differences(view.centers, rotated_translated_centers)
        grad_centers = np.zeros((len(self.centers), 3))
        for p in range(len(self.centers)):
            X1 = np.sum(centers_diffs1[:, p] * np.expand_dims(gaussians1[:, p],1), axis=0)
            rot1 = rot_mat.T @ np.linalg.inv(cov) @ rot_mat
            grad_E1 = ratio**2*self.coeffs[p] * rot1 @ X1
            #grad_E1 = self.coeffs[p] * rot1 @ X1
            X2 = np.sum(centers_diffs2[:, p] * np.expand_dims(gaussians2[:, p], 1), axis=0)
            rot2 = rot_mat.T @ np.linalg.inv(cov)
            grad_E2 = -2 * ratio * self.coeffs[p] * rot2 @ X2
            #grad_E2 = -2 * self.coeffs[p] * rot2 @ X2
            grad_centers[p] = grad_E1 + grad_E2
        self.centers = self.centers - lr * grad_centers
        return energy




def two_by_two_differences(points1, points2):
    """
    points : array containing a set of point, shape (Nb_point, nb_dim)
    compute matricially 2 by 2 distances between points"""
    points_repeated1 = np.repeat(np.expand_dims(points1,axis=0), len(points2), axis=0)
    points_repeated2 = np.repeat(np.expand_dims(points2, axis=0), len(points1), axis=0)
    points_repeated_transpose2 = np.transpose(points_repeated2, (1,0,2))
    diff = points_repeated1 - points_repeated_transpose2
    return np.transpose(diff, (1,0,2))


def random_init(nb_gaussians, nb_dim):
    centers = 2 * (rd.random((nb_gaussians, nb_dim)) - 0.5)
    coeffs =  rd.random(nb_gaussians)
    # centers, coeffs = centers.astype(np.float32), coeffs.astype(np.float32)
    # gm = self.get_untrunc_GM(centers, sigmas, coeffs)
    # save(f'{self.results_folder}/init.tif', gm)
    return centers, coeffs


def arg_max_array(arr):
    arg_ma = np.unravel_index(arr.argmax(), arr.shape)
    ma = arr[arg_ma]
    return np.array(arg_ma), ma


def find_closest_neighbors(points, nb_closest):
    dist_mat = distance_matrix(points, points)
    np.fill_diagonal(dist_mat, 10**5)
    closest_neighbors = np.argsort(dist_mat, axis=1)[:, :nb_closest]
    return dist_mat, closest_neighbors


def distance_matrix(A, B, squared=True):
    """
    Compute all pairwise distances between vectors in A and B.

    Parameters
    ----------
    A : np.array
        shape should be (M, K)
    B : np.array
        shape should be (N, K)

    Returns
    -------
    D : np.array
        A matrix D of shape (M, N).  Each entry in D i,j represnets the
        distance between row i in A and row j in B.

    See also
    --------
    A more generalized version of the distance matrix is available from
    scipy (https://www.scipy.org) using scipy.spatial.distance_matrix,
    which also gives a choice for p-norm.
    """
    M = A.shape[0]
    N = B.shape[0]

    assert A.shape[1] == B.shape[1], f"The number of components for vectors in A \
        {A.shape[1]} does not match that of B {B.shape[1]}!"

    A_dots = (A*A).sum(axis=1).reshape((M,1))*np.ones(shape=(1,N))
    B_dots = (B*B).sum(axis=1)*np.ones(shape=(M,1))
    D_squared =  A_dots + B_dots -2*A.dot(B.T)

    if squared == False:
        zero_mask = np.less(D_squared, 0.0)
        D_squared[zero_mask] = 0.0
        return np.sqrt(D_squared)

    return D_squared


if __name__ == '__main__':
    m = np.array([[1,4,7], [2,1,4], [3,8,2]])
    c = np.array([[4,1,5], [2,8,9]])
    diffs = two_by_two_differences(m, c)
    print('diffs', diffs[0,1, :])

