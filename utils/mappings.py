import numpy as np
import math
import scipy.constants


def confmap2ra(radar_configs, name, radordeg=None):
    """
    Map confidence map to range(m) and angle(deg): not uniformed angle
    :param name: 'range' for range mapping, 'angle' for angle mapping
    :return: mapping grids
    """
    # TODO: add more args for different network settings
    Fs = radar_configs['sample_freq']
    sweepSlope = radar_configs['sweep_slope']
    num_crop = radar_configs['crop_num']
    fft_Rang = radar_configs['ramap_rsize'] + 2*num_crop
    fft_Ang = radar_configs['ramap_asize']
    c = scipy.constants.speed_of_light

    if name == 'range':
        freq_res = Fs / fft_Rang
        freq_grid = np.arange(fft_Rang) * freq_res
        rng_grid = freq_grid * c / sweepSlope / 2
        rng_grid = rng_grid[num_crop:fft_Rang - num_crop]
        return rng_grid

    if name == 'angle':
        # for [-90, 90], w will be [-1, 1]
        w = np.linspace(math.sin(math.radians(radar_configs['ra_min'])),
                        math.sin(math.radians(radar_configs['ra_max'])),
                        radar_configs['ramap_asize'])
        if radordeg is None or radordeg == 'deg':
            agl_grid = np.degrees(np.arcsin(w))  # rad to deg
        elif radordeg == 'rad':
            agl_grid = np.arcsin(w)
        else:
            raise TypeError
        return agl_grid


def labelmap2ra(radar_configs, name):
    """
    Map label map to range(m) and angle(deg): uniformed angle
    :param name: 'range' for range mapping, 'angle' for angle mapping
    :return: mapping grids
    """
    # TODO: add more args for different network settings
    Fs = radar_configs['sample_freq']
    sweepSlope = radar_configs['sweep_slope']
    num_crop = radar_configs['crop_num']
    fft_Rang = radar_configs['ramap_rsize_label'] + 2*num_crop
    fft_Ang = radar_configs['ramap_asize_label']
    c = scipy.constants.speed_of_light

    if name == 'range':
        freq_res = Fs / fft_Rang
        freq_grid = np.arange(fft_Rang) * freq_res
        rng_grid = freq_grid * c / sweepSlope / 2
        rng_grid = rng_grid[num_crop:fft_Rang - num_crop]
        rng_grid = np.flip(rng_grid)
        return rng_grid

    if name == 'angle':
        agl_grid = np.linspace(radar_configs['ra_min_label'], radar_configs['ra_max_label'],
                               radar_configs['ramap_asize_label'])  # rad to deg
        return agl_grid


if __name__ == '__main__':
    w = np.arange(-180, 181, 4)
    print(w.shape)
