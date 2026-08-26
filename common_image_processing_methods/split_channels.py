from manage_files.read_save_files import read_images_in_folder, save, make_dir

pth_in = '/data/eloy/TREx/deconv'
pth_out = f'/data/eloy/TREx/deconv'


images, fn = read_images_in_folder(pth_in)
print('shp', images[0].shape)
nb_channels = 2


for c in range(1, nb_channels+1):
    make_dir(f'{pth_out}/c{c}')

for i in range(len(images)):
    n = images[i].shape[0] // 2
    im1 = images[i][[2*x for x in range(n)]]
    im2 = images[i][[2*x+1 for x in range(n)]]
    save(f'{pth_out}/c1/{fn[i]}', im1)
    save(f'{pth_out}/c2/{fn[i]}', im2)








