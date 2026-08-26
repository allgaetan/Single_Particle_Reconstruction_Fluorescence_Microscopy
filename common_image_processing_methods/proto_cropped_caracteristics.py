from manage_files.read_save_files import read_images_in_folder
from manage_files.paths import *

pth_data = f'{PATH_REAL_DATA}/SAS6/picking/deconv_cropped_proto'
ims, fns = read_images_in_folder(f'{pth_data}/good')
f = open(f'{pth_data}/fns')
print('file names', fns, file=f)


list_fns = []