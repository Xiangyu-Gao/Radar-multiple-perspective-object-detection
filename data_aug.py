import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import scipy.io as spio
from random import randint, random
from utils.mappings import confmap2ra
from utils import find_nearest
from config import radar_configs
from dataPrepare.relocate_dataset import produce_RA_slice


Max_trans_agl = 20
Max_trans_rng = 40
range_grid = confmap2ra(radar_configs, name='range')
angle_grid = confmap2ra(radar_configs, name='angle') # middle number index 63, 64


def resamp_shiftrange(data, shift_range):
    assert len(data.shape) == 5
    data_new = torch.zeros_like(data)
    if shift_range > 0:
        for ir in range(data.shape[3]):
            data_new[:, :, :, ir, :] = resample_range(data[:, :, :, ir, :], ir, shift_range)
    else:
        for ir in range(data.shape[3]):
            data_new[:, :, :, ir, :] = resample_range(data[:, :, :, ir, :], ir - shift_range, shift_range)
    return data_new


def resample_range(data, rangeid, shift_range):
    # rangeid, shift_range can all be id
    # the format of data [batch, C=2, window, angle]
    assert len(data.shape) == 4
    Is_Tensor = False
    if torch.is_tensor(data):
        Is_Tensor = True
        # convert to numpy
        data = data.cpu().detach().numpy()

    # rangeid, shift_range can all be id
    angle_len = len(angle_grid)
    data_new = np.zeros_like(data)
    interval = range_grid[rangeid] * abs(angle_grid[angle_len//2:])
    interval_new = range_grid[rangeid + shift_range] * abs(angle_grid[angle_len//2:])
    start_id = angle_len//2
    # move upwards
    if shift_range > 0:
        for id in range(angle_len//2, angle_len):
            need_id, _ = find_nearest(interval, interval_new[id - angle_len//2])
            start_id = min(start_id, need_id + angle_len//2)
            data_new[:, :, :, id] = np.mean(data[:, :, :, start_id:need_id + 1 + angle_len//2], axis=-1)
            data_new[:, :, :, angle_len-1-id] = np.mean(data[:, :, :, angle_len//2 - need_id - 1:angle_len - start_id],
                                                        axis=-1)
            start_id = need_id + 1 + angle_len//2
            if interval_new[id - angle_len//2] >= interval[-1]:
                break
     # move downwards
    else:
        for id in range(angle_len//2, angle_len):
            need_id, _ = find_nearest(interval_new, interval[id - angle_len//2])
            start_id = min(start_id, need_id + angle_len//2 + 1)
            if id == angle_len//2:
                # interpolate the angle between index 63~64
                data_new[:, :, :, angle_len//2-need_id-1:need_id+angle_len//2+1] = \
                    np.linspace(data[:, :, :, angle_len//2-1], data[:, :, :, angle_len//2], num=2*need_id+2, axis=-1)
                start_id = need_id + 1 + angle_len//2
            else:
                data_new[:, :, :, start_id-1:need_id+angle_len//2+1] = \
                    np.linspace(data[:, :, :, id-1], data[:, :, :, id], num=need_id+angle_len//2-start_id+2, axis=-1)
                data_new[:, :, :, angle_len//2-need_id-1:angle_len-start_id+1] = \
                    np.linspace(data[:, :, :, angle_len-id-1], data[:, :, :, angle_len-id],
                                num=need_id+angle_len//2-start_id+2, axis=-1)
                start_id = need_id + 1 + angle_len//2

            if interval[id - angle_len//2] >= interval_new[-1]:
                break
    if Is_Tensor:
        data_new = torch.from_numpy(data_new)

    return data_new


def resample(data, interval_cur, interval_new):
    Is_Tensor = False
    if torch.is_tensor(data):
        Is_Tensor = True
        # convert to numpy
        data = data.cpu().detach().numpy()
    data_new = np.zeros_like(data)
    interv_len = len(interval_cur)
    max_interv = max(abs(interval_cur))
    start_id_new = 0  # for interval_new
    start_id_cur = 0  # for interval

    for i in range(interv_len):
        if i < start_id_new:
            continue
        # the last element
        if i == interv_len - 1:
            data_new[:, :, :, :, i] = np.mean(data[:, :, :, :, start_id_cur:], axis=-1)
            break

        len_cell_new = interval_new[i+1] - interval_new[i]
        len_cell_cur = interval_cur[start_id_cur+1] - interval_cur[start_id_cur]
        # deal with negative number: 90 degrees, and -90 degrees
        if len_cell_cur < 0:
            len_cell_cur = len_cell_cur + max_interv * 2

        if len_cell_new < len_cell_cur:
            # the interval of new is less than the old interval
            # so we need to find interpolate the interval new
            idn, _ = find_nearest(interval_new - interval_new[start_id_new], len_cell_cur)
            # interpolate the cell between i and idn
            data_new[:, :, :, :, i:idn+1] = np.linspace(data[:, :, :, :, start_id_cur], data[:, :, :, :, start_id_cur+1],
                                                     num=idn+1-i, axis=-1)
            start_id_new = min(idn + 1, interv_len - 1)
            start_id_cur = min(start_id_cur + 1, interv_len - 1)
        else:
            # the interval of new is larger than the old interval
            # so we need to sum multiple cells in old interval
            idn, _ = find_nearest(interval_cur[start_id_cur:] - interval_cur[start_id_cur], len_cell_new)
            # sum the cell between start_id_cur and start_id_cur+idn to cell i
            data_new[:, :, :, :, i] = np.mean(data[:, :, :, :, start_id_cur:start_id_cur+idn+1], axis=-1)

            start_id_cur = min(start_id_cur + idn + 1, interv_len - 1)
            start_id_new = min(start_id_new + 1, interv_len - 1)

    if Is_Tensor:
        data_new = torch.from_numpy(data_new)

    return data_new


def resamp_shiftangle(data, shift_angle, axis=-1):
    # rangeid, shift_range can all be id
    # the format of data [batch, C=2, window, range, angle]
    assert len(data.shape) == 5
    interval_new = angle_grid
    interval = np.roll(interval_new, shift_angle, axis=0)
    if axis == -1 or axis == 4:
        data_new = resample(data, interval, interval_new)
    else:
        data_new = resample(torch.transpose(data, axis, -1), interval, interval_new)
        data_new = torch.transpose(data_new, axis, -1)

    return data_new


def Flip(data, data_va, confmap):
    # flip the angle dimension
    shape = data.shape
    assert len(shape) == 5
    data = torch.flip(data, [4])
    confmap = torch.flip(confmap, [4])
    if data_va is not None:
        data_va = torch.flip(data_va, [3])
        return data, data_va, confmap
    else:
        return data, None, confmap


def transition_angle(data, data_va, confmap, trans_angle=None):
    # shift_angle > 0, move rightward
    # shift_range < 0, move leftward
    if trans_angle is None:
        shift_angle = randint(-Max_trans_agl, Max_trans_agl)
    else:
        shift_angle = trans_angle
    shape = data.shape
    assert len(shape) == 5
    if data_va is not None:
        if shift_angle != 0:
            data_new = torch.roll(data, shift_angle, 4)
            # data_new = resamp_shiftangle(data_new, shift_angle, axis=4)
            data_va_new = torch.roll(data_va, shift_angle, 3)
            # data_va_new = resamp_shiftangle(data_va_new, shift_angle, axis=3)
            confmap_new = torch.roll(confmap, shift_angle, 4)
            # confmap_new = resamp_shiftangle(confmap_new, shift_angle, axis=4)
            return data_new, data_va_new, confmap_new
        else:
            return data, data_va, confmap
    else:
        if shift_angle != 0:
            data_new = torch.roll(data, shift_angle, 4)
            # data_new = resamp_shiftangle(data_new, shift_angle, axis=4)
            confmap_new = torch.roll(confmap, shift_angle, 4)
            # confmap_new = resamp_shiftangle(confmap_new, shift_angle, axis=4)
            return data_new, None, confmap_new
        else:
            return data, None, confmap


def interpolation(data, size=None):
    num_noise_cand = 100
    shape = data.shape
    # print(shape)
    assert len(shape) == 5
    if shape[1] == 2:
        data1 = torch.flatten(data[0, 0, 0:5, :, :])
        data2 = torch.flatten(data[0, 1, 0:5, :, :])
        data_amp = data1 ** 2 + data2 ** 2
        _, indices = torch.sort(data_amp)
        noise_cand1 = np.zeros(num_noise_cand)
        noise_cand2 = np.zeros(num_noise_cand)
        for i, index in enumerate(indices[0:num_noise_cand]):
            noise_cand1[i] = data1[index]
            noise_cand2[i] = data2[index]

        if size is not None:
            need_size = shape[0] * shape[2] * size[0] * size[1]
            noise1 = np.zeros(need_size)
            noise2 = np.zeros(need_size)
            for i in range(need_size):
                noise_id = randint(0, num_noise_cand-1)
                noise1[i] = noise_cand1[noise_id]
                noise2[i] = noise_cand2[noise_id]

            noise1 = np.reshape(noise1, (shape[0], 1, shape[2], size[0], size[1]))
            noise2 = np.reshape(noise2, (shape[0], 1, shape[2], size[0], size[1]))
            noise = np.concatenate((noise1, noise2), axis=1)
        else:
            zero_inds = (data[0, 0, 0, :, :] == 0).nonzero()
            # print(zero_inds)
            need_size = shape[0] * shape[2]
            noise1 = np.zeros(need_size)
            noise2 = np.zeros(need_size)

            for ind in zero_inds:
                for i in range(need_size):
                    noise_id = randint(0, num_noise_cand - 1)
                    noise1[i] = noise_cand1[noise_id]
                    noise2[i] = noise_cand2[noise_id]
                noise1 = np.reshape(noise1, (shape[0], 1, shape[2]))
                noise2 = np.reshape(noise2, (shape[0], 1, shape[2]))
                noise = np.concatenate((noise1, noise2), axis=1)
                data[:, :, :, ind[0], ind[1]] = torch.from_numpy(noise)

    elif shape[1] == 1:
        data1 = torch.flatten(data[0, 0, 0:5, :, :])
        _, indices = torch.sort(data1)
        noise_cand = np.zeros(num_noise_cand)
        for i, index in enumerate(indices[0:num_noise_cand]):
            noise_cand[i] = data1[index]

        if size is not None:
            need_size = shape[0] * shape[2] * size[0] * size[1]
            noise = np.zeros(need_size)
            for i in range(need_size):
                noise[i] = noise_cand[randint(0, num_noise_cand-1)]
            noise = np.reshape(noise, (shape[0], 1, shape[2], size[0], size[1]))
        else:
            zero_inds = (data[0, 0, 0, :, :] == 0).nonzero()
            need_size = shape[0] * shape[2]
            noise = np.zeros(need_size)
            for ind in zero_inds:
                for i in range(need_size):
                    noise_id = randint(0, num_noise_cand - 1)
                    noise[i] = noise_cand[noise_id]
                noise = np.reshape(noise, (shape[0], 1, shape[2]))
                data[:, :, :, ind[0], ind[1]] = torch.from_numpy(noise)

    else:
        print('error')

    if size is not None:
        return noise
    else:
        return data


def transition_range(data, data_rv, confmap, trans_range=None):
    # shift_range > 0, move upward
    # shift_range < 0, move downward
    if trans_range is None:
        shift_range = randint(-Max_trans_rng, Max_trans_rng)
    else:
        shift_range = trans_range
    shape = data.shape
    assert len(shape) == 5
    if data_rv is not None:
        if shift_range != 0:
            data_new = torch.zeros_like(data)
            data_rv_new = torch.zeros_like(data_rv)
            confmap_new = torch.zeros_like(confmap)
            gene_noise_data = torch.from_numpy(interpolation(data, [abs(shift_range), 128]))
            gene_noise_data_rv = torch.from_numpy(interpolation(data_rv, [abs(shift_range), 128]))

            if shift_range > 0:
                compen_mag = np.divide(range_grid[0:shape[3]-shift_range], range_grid[shift_range:shape[3]]) ** 2
                compen_mag = torch.from_numpy(compen_mag).view(1, 1, 1, -1, 1)
                # data_new[:, :, :, shift_range:shape[3], :] = resamp_shiftrange(data[:, :, :, 0:shape[3]-shift_range, :],
                #                                                                shift_range) * compen_mag
                data_new[:, :, :, shift_range:shape[3], :] = data[:, :, :, 0:shape[3] - shift_range, :] * compen_mag
                # data_new[:, :, :, shift_range:shape[3], :] = interpolation(data_new[:, :, :, shift_range:shape[3], :])
                data_new[:, :, :, 0:shift_range, :] = gene_noise_data
                data_rv_new[:, :, :, shift_range:shape[3], :] = data_rv[:, :, :, 0:shape[3] - shift_range, :] * compen_mag
                data_rv_new[:, :, :, 0:shift_range, :] = gene_noise_data_rv
                # confmap_new[:, :, :, shift_range:shape[3], :] = resamp_shiftrange(confmap[:, :, :, 0:shape[3]-shift_range, :],
                #                                                                   shift_range)
                confmap_new[:, :, :, shift_range:shape[3], :] = confmap[:, :, :, 0:shape[3] - shift_range, :]
            else:
                shift_range = abs(shift_range)
                compen_mag = np.divide(range_grid[shift_range:shape[3]], range_grid[0:shape[3]-shift_range]) ** 2
                compen_mag = torch.from_numpy(compen_mag).view(1, 1, 1, -1, 1)
                # data_new[:, :, :, 0:shape[3]-shift_range, :] = resamp_shiftrange(data[:, :, :, shift_range:shape[3], :],
                #                                                                  -shift_range) * compen_mag
                data_new[:, :, :, 0:shape[3] - shift_range, :] = data[:, :, :, shift_range:shape[3], :] * compen_mag
                # data_new[:, :, :, 0:shape[3]-shift_range, :] = interpolation(data_new[:, :, :, 0:shape[3]-shift_range, :])
                data_new[:, :, :, shape[3]-shift_range:shape[3], :] = gene_noise_data
                data_rv_new[:, :, :, 0:shape[3]-shift_range, :] = data_rv[:, :, :, shift_range:shape[3], :] * compen_mag
                data_rv_new[:, :, :, shape[3]-shift_range:shape[3], :] = gene_noise_data_rv
                # confmap_new[:, :, :, 0:shape[3]-shift_range, :] = resamp_shiftrange(confmap[:, :, :, shift_range:shape[3], :],
                #                                                                     -shift_range)
                confmap_new[:, :, :, 0:shape[3] - shift_range, :] = confmap[:, :, :, shift_range:shape[3], :]

            return data_new, data_rv_new, confmap_new

        else:
            return data, data_rv, confmap
    else:
        if shift_range != 0:
            data_new = torch.zeros_like(data)
            confmap_new = torch.zeros_like(confmap)
            gene_noise_data = torch.from_numpy(interpolation(data, [abs(shift_range), 128]))

            if shift_range > 0:
                compen_mag = np.divide(range_grid[0:shape[3] - shift_range], range_grid[shift_range:shape[3]]) ** 2
                compen_mag = torch.from_numpy(compen_mag).view(1, 1, 1, -1, 1)
                # data_new[:, :, :, shift_range:shape[3], :] = resamp_shiftrange(data[:, :, :, 0:shape[3] - shift_range, :],
                #                                                                shift_range) * compen_mag
                data_new[:, :, :, shift_range:shape[3], :] = data[:, :, :, 0:shape[3] - shift_range, :] * compen_mag
                # data_new[:, :, :, shift_range:shape[3], :] = interpolation(data_new[:, :, :, shift_range:shape[3], :])
                data_new[:, :, :, 0:shift_range, :] = gene_noise_data
                # confmap_new[:, :, :, shift_range:shape[3], :] = resamp_shiftrange(confmap[:, :, :, 0:shape[3] - shift_range, :],
                #                                                                   shift_range)
                confmap_new[:, :, :, shift_range:shape[3], :] = confmap[:, :, :, 0:shape[3] - shift_range, :]
            else:
                shift_range = abs(shift_range)
                compen_mag = np.divide(range_grid[shift_range:shape[3]], range_grid[0:shape[3] - shift_range]) ** 2
                compen_mag = torch.from_numpy(compen_mag).view(1, 1, 1, -1, 1)
                # data_new[:, :, :, 0:shape[3] - shift_range, :] = resamp_shiftrange(data[:, :, :, shift_range:shape[3], :],
                #                                                                    -shift_range) * compen_mag
                data_new[:, :, :, 0:shape[3] - shift_range, :] = data[:, :, :, shift_range:shape[3], :] * compen_mag
                # data_new[:, :, :, 0:shape[3] - shift_range, :] = interpolation(data_new[:, :, :, 0:shape[3] - shift_range, :])
                data_new[:, :, :, shape[3] - shift_range:shape[3], :] = gene_noise_data
                # confmap_new[:, :, :, 0:shape[3] - shift_range, :] = resamp_shiftrange(confmap[:, :, :, shift_range:shape[3], :],
                #                                                                       -shift_range)
                confmap_new[:, :, :, 0:shape[3] - shift_range, :] = confmap[:, :, :, shift_range:shape[3], :]
            return data_new, None, confmap_new

        else:
            return data, None, confmap



def Aug_data(data, data_rv, data_va, confmap, type=None):
    if type == 'mix':
        prob = random()
        if prob < 0.3:
            data, data_va, confmap = Flip(data, data_va, confmap)
        prob = random()
        if prob < 0.4:
            data, data_va, confmap = transition_angle(data, data_va, confmap)
        prob = random()
        if prob < 0.4:
            data, data_rv, confmap = transition_range(data, data_rv, confmap)
    else:
        prob = random()
        if prob < 0.2:
            data, data_va, confmap = Flip(data, data_va, confmap)
        elif prob < 0.5:
            data, data_va, confmap = transition_angle(data, data_va, confmap)
        elif prob < 0.8:
            data, data_rv, confmap = transition_range(data, data_rv, confmap)
        else:
            pass

    return data, data_rv, data_va, confmap


# def convert_cpx2pol(data):


if __name__ == '__main__':
    # print(angle_grid)
    # a = np.array([[1, 2, 3], [4, 5, 6]])
    # print(np.linspace(a[0,:], a[1, :], num=5))
    # a = np.ones((128,3,2))
    # b = map_angle(a, 7, 1)
    # print(b)

    # # open a file, where you stored the pickled data
    # file = open('/home/admin-cmmb/Documents/RODNet_dop/data/confmaps_gt/train_all/2019_04_09_bms1000.pkl', 'rb')
    # # dump information to that file
    # all_confmap = pickle.load(file)
    # # close the file
    # file.close()
    # confmap = all_confmap[0][0][0:3]
    # confmap = np.reshape(confmap, (1, 3, 1, 128, 128))
    # print(confmap.shape)
    data = np.load('/mnt/sda/3DRadardata/2019_04_09/2019_04_09_bms1000/RA_NPY/0000/000000.npy')
    data_amp = data[:, :, 0]**2 + data[:, :, 1]**2
    plt.figure()
    plt.imshow(data_amp, vmax=0.01)
    plt.gca().invert_yaxis()
    # plt.show()
    #
    # data = np.transpose(data, (2, 0, 1))
    # # print(data.shape)
    # data = np.reshape(data, (1, 2, 1, 128, 128))
    # # print(data.shape)
    # data = torch.from_numpy(data)
    # confmap = torch.from_numpy(confmap)
    # data, _, confmap = transition_range(data, None, confmap, trans_range=-20)
    # # data, _, confmap = transition_angle(data, None, confmap, trans_angle=None)
    # data_amp1 = data[0, 0, 0, :, :]**2 + data[0, 1, 0, :, :]**2
    # plt.figure()
    # plt.imshow(data_amp1, vmax=0.01)
    # plt.gca().invert_yaxis()
    # plt.show()

    mat_data_dir = '/media/admin-cmmb/Elements/CRdataset/2019_04_09/2019_04_09_bms1000/WIN_R_MAT/2019_04_09_bms1000_000003.mat'
    mat = spio.loadmat(mat_data_dir, squeeze_me=True)
    data = np.asarray(mat["R_data"])
    # shift_range
    shift_range = 1
    shift_angle = -30 # in degrees
    shift_angle = math.radians(shift_angle)
    compen_agl = np.arange(8) * shift_angle
    compen_agl = np.reshape(compen_agl, (1, -1, 1))
    data_new = np.zeros_like(data)
    if shift_range > 0:
        compen_mag = np.divide(range_grid[0:128 - shift_range], range_grid[shift_range:128]) ** 2
        compen_mag = np.reshape(compen_mag, (-1, 1, 1))
        compen_pha = np.divide(range_grid[0:128 - shift_range], range_grid[shift_range:128])
        compen_pha = np.reshape(compen_pha, (-1, 1, 1))
        data_new[shift_range:128, :, :] = data[0:128 - shift_range, :, :] * compen_mag
        # compensate the phase change
        angle = np.angle(data_new[shift_range:128, :, :])
        angle[angle < 0] += 2 * math.pi
        angle = angle * compen_pha
        angle = angle + compen_agl
        amp = np.abs(data_new[shift_range:128, :, :])
        data_new[shift_range:128, :, :] = amp * np.cos(angle) + amp * np.sin(angle) * 1j

    elif shift_range < 0:
        shift_range = abs(shift_range)
        compen_mag = np.divide(range_grid[shift_range:128], range_grid[0:128 - shift_range]) ** 2
        compen_mag = np.reshape(compen_mag, (-1, 1, 1))
        compen_pha = np.divide(range_grid[shift_range:128], range_grid[0:128 - shift_range])
        compen_pha = np.reshape(compen_pha, (-1, 1, 1))
        data_new[0:128 - shift_range, :, :] = data[shift_range:128, :, :] * compen_mag
        angle = np.angle(data_new[0:128 - shift_range, :, :])
        angle[angle < 0] += 2 * math.pi
        angle = angle * compen_pha
        angle = angle + compen_agl
        amp = np.abs(data_new[0:128 - shift_range, :, :])
        data_new[0:128 - shift_range, :, :] = amp * np.cos(angle) + amp * np.sin(angle) * 1j

    ra_data = produce_RA_slice(data_new)
    data_amp = ra_data[:, :, 0, 0] ** 2 + ra_data[:, :, 0, 1] ** 2
    plt.figure()
    plt.imshow(data_amp, vmax=0.01)
    plt.gca().invert_yaxis()
    # print(data_new.shape)
    # print(ra_data.shape)
    plt.show()