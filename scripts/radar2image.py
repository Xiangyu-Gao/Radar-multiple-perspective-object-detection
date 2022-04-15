import numpy as np 
import os
import math
import scipy.io as spio
# import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import scipy.constants

Fs = 4e6
sweepSlope = 21.0017e12
c = scipy.constants.speed_of_light
num_crop = 3

def create_angle_grid():

    w = np.linspace(-1, 1, 128)
    agl_grid = np.degrees(np.arcsin(w))  # rad to deg
    # print(agl_grid.shape)
    return agl_grid

def create_range_grid(fft_Rang):

    freq_res = Fs / fft_Rang
    freq_grid = np.arange(fft_Rang) * freq_res
    rng_grid = freq_grid * c / sweepSlope / 2
    rng_grid = rng_grid[num_crop:fft_Rang - num_crop]

    return rng_grid


def main():
    source_dir = '/mnt/nas_crdataset/'
    dest_dir = '/mnt/nas_crdataset2/'

    processd_date = ['2019_09_29']
    seqs_special = ['2019_09_29_onrd000', '2019_09_29_onrd001', '2019_09_29_onrd003', '2019_09_29_onrd004',
                    '2019_09_29_onrd017', '2019_09_29_onrd018']

    # step2: make angle mapping to the range [-60, 60]
    agl_grid = create_angle_grid()
    new_agl_grid = np.arange(-60, 61, 1)
    # print(new_agl_grid)
    # print(agl_grid)
    new_rng_grid = create_range_grid(128)
    old_rng_grid = create_range_grid(134)
    # print(old_rng_grid.shape)
    # print(new_rng_grid.shape)

    for sub_folder in processd_date:
        sub_source_dir = source_dir + sub_folder + '/'
        sub_dest_dir = dest_dir + sub_folder + '/'

        for items in sorted(os.listdir(sub_source_dir)):
            if items not in seqs_special:
                continue

            item_source_dir = sub_source_dir + items + '/' + 'WIN_PROC_MAT_DATA'
            item_dest_dir = sub_dest_dir + items + '/' + 'WIN_RADAR_LABEL_IMAGE'
            num_item_dest = len(os.listdir(sub_dest_dir + items + '/' + 'dets_refine'))

            if os.path.exists(item_source_dir):
                if not os.path.exists(item_dest_dir):
                    os.makedirs(item_dest_dir)

                for files in sorted(os.listdir(item_source_dir))[-num_item_dest:]:
                    src = os.path.join(item_source_dir,files)
                    dst = os.path.join(item_dest_dir,files)
                    dst = dst.replace(".mat", ".jpg")

                    mat = spio.loadmat(src, squeeze_me = True)
                    data = np.abs(mat["Angdata_crop"])
                    # print(data.shape)
                    # print(src)

                    # step1: absolute and average across different chirps
                    data_aveg = np.average(data, axis = 2)
                    # print(data_aveg.shape)

                    data_map = np.zeros([122,121])
                    for r_count,r_elem in enumerate(new_rng_grid):
                        r_index = np.argmin(np.abs(old_rng_grid - r_elem))
                        for a_count, elem in enumerate(new_agl_grid):
                            a_index = np.argmin(np.abs(agl_grid - elem))
                            # print(r_count,a_count)
                            data_map[r_count, a_count] = data_aveg[r_index, a_index]
                    # print(data_map)
                    # print(data_map.shape)

                    # step3: normalize to 0~255
                    max_value = 2
                    data_norm = 255*data_map/max_value
                    data_norm = np.clip(data_norm, 0, 255) # clip
                    data_norm = np.flipud(data_norm) # flip along the first dimension
                    # print(data_/norm)

                    # step4: save files
                    img = data_norm.astype(np.uint8)
                    matplotlib.image.imsave(dst, img)
                    # imgplot = plt.imshow(img)
                    # plt.show()

                    print('finished', dst)

        print('finished', item_source_dir)


if __name__ == "__main__":
    main()
