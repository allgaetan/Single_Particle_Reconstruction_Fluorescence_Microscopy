import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import imageio
import mrcfile
import time


"""

This code takes as input a folder containing 2D crops of the Nuclear Pore Complex (NPC). 
Each crop is separated in 8 sectors and the number of activated sectors is counted. 
Then the histogram of the number of activated sectors is saved.

"""

"""Variables to specify"""

#npc_path = '/home/eloy/ScipionUserData/projects/recons/Runs/000652_EmanProtTomoExtraction/small_npc' # chemin des images d'entrée
#save_fold = '/home/eloy/Documents/stage_reconstruction_spfluo/NPC_quantify_symmetry' # chemin du fichier de sauvegarde
npc_path = '3im_npc_c8'  # input folder, contaning a set of 2D NPC crops
save_fold = '3im_npc_c8_res' # output folder. In this folder are saved the separation in 8 sectors for each NPC crop,
# as well as the histogram of the number of activated sectors

symmetryC = 8 # symmetry of the object (8 for the NPC)
thershold_conversion_to_pc = 100 # The image is converted as a point cloud, defined as the set of pixel coordinates higher than a given threshold
alpha = 0.90 # We keep a proportion alpha of the nearest points from the center
thersh_activated_sectors = 1/symmetryC * 0.2 # A sector is activated if the total number of points that it contains
                                             # is greater than 'thersh_activated_sectors' times the total number of points in the image


def read_image(path, mrc=False):
    if not mrc:
        return np.array(imageio.mimread(path, memtest=False))
    else:
        print('path', path)
        return mrcfile.read(path)


def read_images_in_folder(fold, alphabetic_order=True, mrc=False):
    """read all the images inside folder fold"""
    files = os.listdir(fold)
    if alphabetic_order:
        files = sorted(files)
    images = []
    files_without_dir = []
    for fn in files:
        pth = f'{fold}/{fn}'
        if not os.path.isdir(pth):
            print('pth', pth)
            im = read_image(pth, mrc)
            images.append(np.squeeze(im))
            files_without_dir.append(fn)

    return images, files_without_dir


def convert_im_to_point_cloud(im, thesh):
    coordinates = np.where(im>=thesh)
    coordinates = np.array(coordinates).T
    return coordinates


def write_array_csv(np_array, path):
    pd.DataFrame(np_array).to_csv(path)


def angular_diff(target, source):
    a = target - source
    a = np.mod(a + np.pi, 2*np.pi) - np.pi
    return a


def make_dir(dir):
    """creates folder at location dir if i doesn't already exist"""
    if not os.path.exists(dir):
        print(f'directory {dir} created')
        os.makedirs(dir)



make_dir(save_fold)

imgs, fns = read_images_in_folder(npc_path, mrc=False)


histo = np.zeros(symmetryC+1)
for i in range(len(imgs)):
    t = time.time()
    save_fold_im = f'{save_fold}/{fns[i]}'
    make_dir(save_fold_im)
    img = imgs[i]
    size = img.shape[1]
    print('size', size)
    #plot_ortho([img3d])
    #img = img3d[size//2, :, :] # on garde la coupe centrale de l'image
    pc = convert_im_to_point_cloud(img, thershold_conversion_to_pc) # conversion en nuages de points
    pc = 2*pc/size - 1
    barycenter = np.mean(pc, axis=0)
    pc = pc - barycenter
    # Conversion into polar coordinates
    radiuses = np.linalg.norm(pc, axis=1)
    angles = np.arctan2(pc[:, 0], pc[:, 1])

    # Suppression of the farthest points from the center
    count, bins = np.histogram(radiuses, bins=50)
    repart_fun = [np.sum(count[:i]) for i in range(len(count))]/np.sum(count)
    idx_radius = np.min(np.where(repart_fun >= alpha))
    radius_npc = bins[idx_radius]
    radiuses_selected = radiuses[np.where(radiuses<=radius_npc)]
    angles_selected = angles[np.where(radiuses<=radius_npc)]

    # Determine the orientation of the NPC and find the boarders of the symmetryC=8 sectors
    angles_mod = np.mod(angles_selected, 2*np.pi/symmetryC)
    angles_grid = np.arange(0, 2*np.pi/symmetryC, 0.01)
    angles_mod_repeated = np.repeat(np.expand_dims(angles_mod, axis=0), len(angles_grid), axis=0)
    angles_grid_repeated = np.repeat(np.expand_dims(angles_grid, axis=1), len(angles_mod), axis=1)
    diff = np.abs(angular_diff(angles_grid_repeated, angles_mod_repeated))
    summ_diff = np.sum(diff, axis=1)
    orientation_npc = angles_grid[np.argmin(summ_diff)]
    sectors = np.array([orientation_npc - np.pi/symmetryC + 2*np.pi/symmetryC*k for k in range(symmetryC)])

    # Plot the circle and sectors for visualization
    radius_npc_pixel = radius_npc * size/2
    circle_points = np.array([[radius_npc_pixel * np.cos(theta) + size//2, radius_npc_pixel * np.sin(theta) + size//2] for theta in np.arange(0, 2*np.pi, 0.01)])
    sectors_cartesian = np.array([[radius_npc_pixel * np.cos(theta) + size//2, radius_npc_pixel * np.sin(theta) + size//2] for theta in sectors])
    point_size = 5
    plt.imshow(img)
    plt.scatter(circle_points[:, 0], circle_points[:, 1], c='red', s=point_size)
    for i in range(len(sectors_cartesian)):
        a = sectors_cartesian[i, 0]
        b = sectors_cartesian[i, 1]
        x = np.linspace(size//2, a, 1000)
        y = np.linspace(size//2, b, 1000)
        plt.scatter(x, y, c='red', s=point_size)
    plt.savefig(f'{save_fold_im}/sectors_visu.png')
    plt.close()
    # counts the number of points per sector
    count_per_sectors = np.zeros(symmetryC)
    for a in angles_selected:
        a_mod = np.mod(a, 2*np.pi)
        for i in range(len(sectors)):
            start = np.mod(sectors[i], 2*np.pi)
            end = np.mod(sectors[i+1], 2*np.pi) if i != len(sectors) - 1 else np.mod(sectors[0], 2*np.pi) + 2*np.pi
            if a_mod >= start and a_mod < end:
                count_per_sectors[i] += 1
    nb_points = len(angles_selected)
    count_per_sectors_normalized = count_per_sectors/nb_points
    write_array_csv(count_per_sectors_normalized, f'{save_fold_im}/count_per_sectors.csv')
    nb_activated_sectors = len(count_per_sectors_normalized[count_per_sectors_normalized >= thersh_activated_sectors])
    histo[nb_activated_sectors] += 1
    print('time', time.time() - t)

write_array_csv(histo, f'{save_fold}/histo_activated_sectors.csv')
plt.bar([str(i) for i in range(symmetryC+1)], histo)
plt.savefig(f'{save_fold}/histo_activated_secotrs.png')
plt.close()





#plot_ortho([img,img_back])





