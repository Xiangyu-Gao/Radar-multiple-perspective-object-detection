import os
import numpy as np
import pickle
import argparse

from utils.confidence_map import generate_confmap, normalize_confmap, add_noise_channel
from utils.read_annotations import read_ra_labels_csv
from utils.visualization import visualize_confmap

from config import train_sets, test_sets, valid_sets
from config import n_class, radar_configs


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare RODNet data.')
    parser.add_argument('-m', '--mode', type=str, dest='mode', help='choose from train, valid, test, supertest')
    parser.add_argument('-dd', '--datadir', type=str, dest='data_dir', default='./data',
                        help='data directory to save the prepared data')
    args = parser.parse_args()
    return args


def prepare_data(sets, set_type='train', viz=False):
    root_dir_npy = './template_files/train_test_data/'
    sets_dates = sets['dates']
    sets_seqs = sets['seqs']

    for dateid in range(len(sets_dates)):
        seqs = sets_seqs[dateid]
        for seq in seqs:
            detail_list = [[], 0]
            confmap_list = [[], []]
            n_data = 900
            # create paths for data
            for fid in range(n_data):
                if radar_configs['data_type'] == 'RISEP' or radar_configs['data_type'] == 'APSEP':
                    path = os.path.join(root_dir_npy, sets_dates[dateid], seq, chirp_folder_name, "%04d", "%06d.npy")
                else:
                    raise ValueError
                detail_list[0].append(path)

            # use labelled RAMap
            seq_path = os.path.join(root_dir_npy, sets_dates[dateid], seq)
            try:
                obj_info_list = read_ra_labels_csv(seq_path)
            except Exception as e:
                print("Load sequence %s failed!" % seq_path)
                print(e)
                continue
            assert len(obj_info_list) == n_data

            for obj_info in obj_info_list:
                confmap_gt = np.zeros((n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                                      dtype=float)
                confmap_gt[-1, :, :] = 1.0
                if len(obj_info) != 0:
                    confmap_gt = generate_confmap(obj_info)
                    confmap_gt = normalize_confmap(confmap_gt)
                    confmap_gt = add_noise_channel(confmap_gt)
                assert confmap_gt.shape == (n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])
                if viz:
                    visualize_confmap(confmap_gt)
                confmap_list[0].append(confmap_gt)
                confmap_list[1].append(obj_info)
                # end objects loop
            confmap_list[0] = np.array(confmap_list[0])

            dir2 = os.path.join(confmap_dir, set_type)
            dir3 = os.path.join(detail_dir, set_type)
            if not os.path.exists(dir2):
                os.makedirs(dir2)
            if not os.path.exists(dir3):
                os.makedirs(dir3)

            # save pkl files
            pickle.dump(confmap_list, open(os.path.join(dir2, seq + '.pkl'), 'wb'))
            # save pkl files
            pickle.dump(detail_list, open(os.path.join(dir3, seq + '.pkl'), 'wb'))

            # end frames loop
        # end seqs loop
    # end dates loop


if __name__ == "__main__":
    """
    Example:
        python prepare_data.py -m train -dd './data/'
    """
    args = parse_args()
    modes = args.mode.split(',')
    data_dir = args.data_dir

    if radar_configs['data_type'] == 'RI':
        chirp_folder_name = 'radar_chirps_RI'
    elif radar_configs['data_type'] == 'AP':
        chirp_folder_name = 'radar_chirps_AP'
    elif radar_configs['data_type'] == 'RISEP':
        chirp_folder_name = 'RA_NPY'
    elif radar_configs['data_type'] == 'APSEP':
        chirp_folder_name = 'radar_chirps_win_APSEP'
    else:
        raise ValueError

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    confmap_dir = os.path.join(data_dir, 'confmaps_gt')
    detail_dir = os.path.join(data_dir, 'data_details')

    for mode in modes:
        if mode == 'train':
            print('Preparing %s sets ...' % mode)
            prepare_data(train_sets, set_type=mode, viz=False)
        elif mode == 'valid':
            print('Preparing %s sets ...' % mode)
            prepare_data(valid_sets, set_type=mode, viz=False)
        elif mode == 'test':
            print('Preparing %s sets ...' % mode)
            prepare_data(test_sets, set_type=mode, viz=False)
        else:
            print("Warning: unknown mode %s" % mode)
