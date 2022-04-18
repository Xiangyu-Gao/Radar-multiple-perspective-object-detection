import scipy.io
import os
import numpy as np
import math
import time
import argparse
from multiprocessing import Pool

from config import data_sets, train_sets, valid_sets, test_sets
from config import radar_configs, rodnet_configs
from utils import str2bool
from utils.visualization import visualize_radar_chirp, visualize_radar_chirps

USE_LOCAL = False

# Example:
#   python preprocess_radar.py -p 1 -np 10


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess radar data.')
    parser.add_argument('-p', '--para', type=str2bool, dest='para', default=False, help='run parallel or not')
    parser.add_argument('-np', '--paranum', type=int, dest='n_para', default=2, help='number of parallel process')
    parser.add_argument('-nc', '--parachirpnum', type=int, dest='n_para_chirp', default=255,
                        help='number of chirps to process at once')
    parser.add_argument('-vb', '--verbose', type=str2bool, dest='verbose', default=False,
                        help='show log information or not')
    args = parser.parse_args()
    return args


def preprocess_radar_data(path, seq_name):
    """
    Pre-process radar data from MAT to NPY, each chirp in one file
    :param path: data sequence path
    :param seq_name: sequence name
    :return: None
    """
    try:
        start_time = time.time()
        k_range = math.ceil(n_chirps / N_PARA_CHIRP)  # number of loops
        for loop in range(k_range):
            # each loop include N_PARA_CHIRP (except last loop)
            radar_mat_names = sorted(os.listdir(os.path.join(path, rodnet_configs['data_folder'])))
            n_radar_frames = len(radar_mat_names)
            start_id = int(float(radar_mat_names[0].split('.')[0].split('_')[-1]))
            if RADAR_DATA_TYPE == 'RI' or RADAR_DATA_TYPE == 'AP':
                temp_mat = np.zeros((N_PARA_CHIRP, n_radar_frames, ramap_rsize, ramap_asize, 2), dtype=np.float32)
            else:
                temp_mat = np.zeros((ramap_rsize, ramap_asize, 2), dtype=np.float32)
            print('Processing seq: %s, loop: %d' % (seq_name, loop))
            for frameid, real_id in enumerate(range(start_id, start_id + n_radar_frames)):
                # print(frameid, real_id)
                if not os.path.exists(os.path.join(path, folder_name)):
                    os.makedirs(os.path.join(path, folder_name))
                mat_name = os.path.join(path, rodnet_configs['data_folder'], '%s_%06d.mat' % (seq_name, real_id))
                mat = scipy.io.loadmat(mat_name)
                for chirpid in range(N_PARA_CHIRP):
                    chirpid_abs = loop * N_PARA_CHIRP + chirpid
                    if LOG_INFO and frameid == 0:
                        print('process chirp', chirpid_abs)
                    if loop * N_PARA_CHIRP + chirpid >= n_chirps:
                        break
                    if RADAR_DATA_TYPE == 'RI':
                        temp_mat[chirpid, frameid, :, :, 0] = np.real(mat['Angdata_crop'][:, :, chirpid_abs])
                        temp_mat[chirpid, frameid, :, :, 1] = np.imag(mat['Angdata_crop'][:, :, chirpid_abs])
                    elif RADAR_DATA_TYPE == 'AP':
                        temp_mat[chirpid, frameid, :, :, 0] = np.absolute(mat['Angdata_crop'][:, :, chirpid_abs])
                        temp_mat[chirpid, frameid, :, :, 1] = np.angle(mat['Angdata_crop'][:, :, chirpid_abs]) \
                                                                   / (2 * math.pi) + 0.5  # normalize to [-1, 1]
                    elif RADAR_DATA_TYPE == 'RISEP':
                        if not os.path.exists(os.path.join(path, folder_name, '%04d' % chirpid_abs)):
                            os.makedirs(os.path.join(path, folder_name, '%04d' % chirpid_abs))
                        temp_mat[:, :, 0] = np.real(mat['Angdata_crop'][:, :, chirpid_abs])
                        temp_mat[:, :, 1] = np.imag(mat['Angdata_crop'][:, :, chirpid_abs])
                        np.save(os.path.join(path, folder_name, '%04d' % chirpid_abs, '%06d.npy' % frameid), temp_mat)
                    elif RADAR_DATA_TYPE == 'APSEP':
                        if not os.path.exists(os.path.join(path, folder_name, '%04d' % chirpid_abs)):
                            os.makedirs(os.path.join(path, folder_name, '%04d' % chirpid_abs))
                        temp_mat[:, :, 0] = np.absolute(mat['Angdata_crop'][:, :, chirpid_abs])
                        temp_mat[:, :, 1] = np.angle(mat['Angdata_crop'][:, :, chirpid_abs]) \
                                                                   / (2 * math.pi) + 0.5  # normalize to [-1, 1]
                        np.save(os.path.join(path, folder_name, '%04d' % chirpid_abs, '%06d.npy' % frameid), temp_mat)

            if RADAR_DATA_TYPE == 'RI' or RADAR_DATA_TYPE == 'AP':
                for j in range(N_PARA_CHIRP):
                    chirpid_abs = loop * N_PARA_CHIRP + j
                    if loop * N_PARA_CHIRP + j >= n_chirps:
                        break
                    if LOG_INFO:
                        print('save chirp', chirpid_abs)
                    np.save(os.path.join(path, folder_name, '%04d.npy' % (chirpid_abs)), temp_mat[j, :, :, :, :])

        print('Seq %s preprocessing finished in %s seconds.' % (seq_name, time.time() - start_time))

    except Exception as e:
        print("Preprocess %s failed: %s" % (seq_name, e))


def debug_preprocess(path, chirpid):
    if chirpid is not None:
        chirp = np.load(os.path.join(path, folder_name, '%04d.npy' % (chirpid)))
        n_frames = chirp.shape[0]
        for frame_id in range(n_frames):
            visualize_radar_chirp(chirp[frame_id])
    else:
        chirps = []
        for i in range(radar_configs['n_chirps']):
            chirp = np.load(os.path.join(path, folder_name, '%04d.npy' % (i)))
            chirps.append(chirp)
        n_frames = chirp.shape[0]
        chirps = np.array(chirps)
        for frame_id in range(n_frames):
            visualize_radar_chirps(chirps[:, frame_id, :, :, :])


if __name__ == "__main__":
    args = parse_args()
    PARA = args.para
    RARA_NUM = args.n_para
    N_PARA_CHIRP = args.n_para_chirp    # number of chirps to process at once
                                        # decrease this number for lower memory usage
    LOG_INFO = args.verbose

    root_dir = data_sets['root_dir']
    sets_dates = data_sets['dates']
    ramap_rsize = radar_configs['ramap_rsize']
    ramap_asize = radar_configs['ramap_asize']
    n_chirps = radar_configs['n_chirps']
    RADAR_DATA_TYPE = radar_configs['data_type']  # 'RI': real + imaginary, 'AP': amplitude + phase

    if USE_LOCAL:
        # set direcotries
        root_dir = "/mnt/disk2/CR_dataset"
        sets_dates = ["2019_07_25"]
        # set parameters
        ramap_rsize = 128
        ramap_asize = 91
        n_chirps = 255

        print('Using local parameters')

    # prepare data paths
    train_sets_full_paths = []
    for i in range(len(sets_dates)):
        seqs = sorted(os.listdir(os.path.join(root_dir, sets_dates[i])))
        train_sets_full_paths.append(seqs)

    train_sets_full_paths = data_sets['seqs']
    paths_tuple = []
    for i in range(len(sets_dates)):
        for train_set in train_sets_full_paths[i]:
            path = os.path.join(root_dir, sets_dates[i], train_set)
            paths_tuple.append((path, train_set))

    if RADAR_DATA_TYPE == 'RI':
        folder_name = 'radar_chirps_RI'
    elif RADAR_DATA_TYPE == 'AP':
        folder_name = 'radar_chirps_AP'
    elif RADAR_DATA_TYPE == 'RISEP':
        folder_name = 'radar_chirps_win_RISEP'
    elif RADAR_DATA_TYPE == 'APSEP':
        folder_name = 'radar_chirps_win_APSEP'
    else:
        raise ValueError

    # separate radar data into different chirps
    if PARA:
        # use multiple threads
        pool = Pool(processes=RARA_NUM)
        pool.starmap(preprocess_radar_data, paths_tuple)
    else:
        # use single thread
        for path, train_set in paths_tuple:
            preprocess_radar_data(path, train_set)

    # for path, train_set in paths_tuple:
    #     debug_preprocess(path, chirpid=0)
