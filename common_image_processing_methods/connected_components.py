
import numpy as np
import matplotlib.pyplot as plt
from manage_files.read_save_files import read_image, save
import mains.synthetic_data.main_pixel_representation_synthetic_data
import cc3d
from common_image_processing_methods.others import normalize


if __name__ == '__main__':
    pth1 = '/home/eloy/Documents/stage_reconstruction_spfluo/real_data/U-ExM/data/raw/results/c1_crop_60_2/recons.tif'
    #pth1 = '/home/eloy/Documents/stage_reconstruction_spfluo/real_data/U-ExM/data/raw/results/c1_crop_60_1/recons_shifted.tif'
    im = read_image(pth1)
    im = normalize(im)
    labels_in = 1 * (im>0.4)
    save('cc.tif', labels_in)
    labels_out, N = cc3d.connected_components(labels_in, return_N=True)
    print(N)

