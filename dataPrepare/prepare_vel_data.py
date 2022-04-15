import os
import sys
import shutil
import numpy as np
import pickle
import argparse

from utils.confidence_map import generate_confmap, normalize_confmap, add_noise_channel
from utils.dataset_tools import fix_cam_drop_frames, calculate_frame_offset
from utils.visualization import visualize_confmap

from config import train_sets, test_sets, valid_sets, supertest_sets
from config import n_class, class_ids, camera_configs, radar_configs, rodnet_configs


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare RODNet data.')
    parser.add_argument('-m', '--mode', type=str, dest='mode', help='choose from train, valid, test, supertest')
    parser.add_argument('-dd', '--datadir', type=str, dest='data_dir', default='./vel_data',
                        help='data directory to save the prepared data')
    args = parser.parse_args()
    return args


def read_vel_label(seq_path, read_frame_idx):
    label = open(os.path.join(seq_path, 'v_label.txt'))
    obj_info = []
    for line in label:
        # for each object
        # frame_id, range_id, angle_id, class_id, doppler_id
        line = line.rstrip().split()
        if int(line[0]) == read_frame_idx:
            # rng_odx, agl_idx, dop_idx, class_iid
            obj_info.append([int(line[1]), int(line[2]), int(line[4]), int(line[3])])

    return obj_info



def prepare_data(sets, set_type='train', viz=False, cam_drop_frame=False):
    root_dir = sets['root_dir']
    root_dir_npy = '/mnt/sda/3DRadardata'

    Label_dir = '/mnt/sda/Labels'
    sets_dates = sets['dates']
    sets_seqs = sets['seqs']
    cam_anno = sets['cam_anno']

    # if set_type == 'train' or set_type == 'valid' or set_type == 'test' or set_type == 'supertest':
    #     if os.path.exists(os.path.join(detail_dir, set_type)):
    #         shutil.rmtree(os.path.join(detail_dir, set_type))
    #     os.makedirs(os.path.join(detail_dir, set_type))
    #     if os.path.exists(os.path.join(confmap_dir, set_type)):
    #         shutil.rmtree(os.path.join(confmap_dir, set_type))
    #     os.makedirs(os.path.join(confmap_dir, set_type))
    # else:
    #     raise ValueError

    for dateid in range(len(sets_dates)):
        if sets_seqs[dateid] is None:
            # if None in some dates, include all the sequences
            seqs = sorted(os.listdir(os.path.join(root_dir, sets_dates[dateid])))
        else:
            seqs = sets_seqs[dateid]
        for seq in seqs:
            seq_path = os.path.join(root_dir, sets_dates[dateid], seq)
            label_seq_path = os.path.join(Label_dir, sets_dates[dateid], seq)
            print(seq_path)
            radar_mat_names = sorted(os.listdir(os.path.join(seq_path, rodnet_configs['data_folder'])))
            n_data = len(radar_mat_names)
            ra_frame_offset = calculate_frame_offset(os.path.join(seq_path, 'start_time.txt'))[0]
            start_id = int(float(radar_mat_names[0].split('.')[0].split('_')[-1]))
            if ra_frame_offset > 0 and cam_anno[dateid]:
                ra_frame_offset += 40
                n_data -= ra_frame_offset
                start_id = ra_frame_offset

            confmap_list = [[], []]
            # print(n_data + start_id)
            # print(start_id)

            for real_id in range(start_id, n_data + start_id):
                sys.stdout.write('\r' + 'processing cam anno %s %04d/%04d' % (seq_path, real_id, n_data + start_id))
                # for each frame
                if 'refine' in rodnet_configs['label_folder']:
                    # read annotations
                    obj_info = read_vel_label(label_seq_path, real_id)
                    # print(obj_info)

                confmap_gt_rv = np.zeros((n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_vsize']),
                                      dtype=float)
                confmap_gt_av = np.zeros((n_class + 1, radar_configs['ramap_asize'], radar_configs['ramap_vsize']),
                                      dtype=float)
                if len(obj_info) != 0:
                    confmap_gt_rv = generate_confmap(obj_info, type='rv')
                    confmap_gt_rv = normalize_confmap(confmap_gt_rv)
                    confmap_gt_rv = add_noise_channel(confmap_gt_rv)
                    confmap_gt_av = generate_confmap(obj_info, type='av')
                    confmap_gt_av = normalize_confmap(confmap_gt_av)
                    confmap_gt_av = add_noise_channel(confmap_gt_av)
                assert confmap_gt_rv.shape == (n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_vsize'])
                assert confmap_gt_av.shape == (n_class + 1, radar_configs['ramap_asize'], radar_configs['ramap_vsize'])

                if viz:
                    visualize_confmap(confmap_gt_rv)
                confmap_list[0].append(confmap_gt_rv)
                confmap_list[1].append(confmap_gt_av)
                # end objects loop
            # format of confmap_list: rv, av
            confmap_list[0] = np.array(confmap_list[0])
            confmap_list[1] = np.array(confmap_list[1])

            # save pkl files
            # check the number of files
            ra_confmap_list = pickle.load(open(os.path.join('./data/confmaps_gt/train_all', seq) + '.pkl', 'rb'))
            assert ra_confmap_list[0].shape == confmap_list[0].shape
            assert ra_confmap_list[0].shape == confmap_list[1].shape
            # print(ra_confmap_list[0].shape, confmap_list[0].shape)
            # input()
            pickle.dump(confmap_list, open(os.path.join(confmap_dir, set_type, seq + '.pkl'), 'wb'))


if __name__ == "__main__":
    """
    Example:
        python prepare_data.py -m train -dd /mnt/ssd2/rodnet/data_refine
    """
    args = parse_args()
    modes = args.mode.split(',')
    data_dir = args.data_dir

    if radar_configs['data_type'] == 'RI':
        chirp_folder_name = 'radar_chirps_RI'
    elif radar_configs['data_type'] == 'AP':
        chirp_folder_name = 'radar_chirps_AP'
    elif radar_configs['data_type'] == 'RISEP':
        # chirp_folder_name = 'radar_chirps_win_RISEP'
        chirp_folder_name = 'RA_NPY'
    elif radar_configs['data_type'] == 'APSEP':
        chirp_folder_name = 'radar_chirps_win_APSEP'
    else:
        raise ValueError

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    confmap_dir = os.path.join(data_dir, 'confmaps_gt')
    if not os.path.exists(confmap_dir):
        os.makedirs(confmap_dir)

    for mode in modes:
        if mode == 'train':
            print('Preparing %s sets ...' % mode)
            prepare_data(train_sets, set_type=mode, viz=False, cam_drop_frame=True)
        elif mode == 'valid':
            print('Preparing %s sets ...' % mode)
            prepare_data(valid_sets, set_type=mode, viz=False, cam_drop_frame=True)
        elif mode == 'test':
            print('Preparing %s sets ...' % mode)
            prepare_data(test_sets, set_type=mode, viz=False, cam_drop_frame=True)
        elif mode == 'supertest':
            print('Preparing %s sets ...' % mode)
            prepare_data(supertest_sets, set_type=mode, viz=False, cam_drop_frame=True)
        else:
            print("Warning: unknown mode %s" % mode)

    # print('Preparing training sets ...')
    # prepare_data(train_sets, set_type='train', viz=False, cam_drop_frame=True)
    # print('Preparing validation sets ...')
    # prepare_data(valid_sets, set_type='valid', viz=False, cam_drop_frame=True)
    # print('Preparing test sets ...')
    # prepare_data(test_sets, set_type='test', viz=False, cam_drop_frame=True)
    # print('Preparing super test sets ...')
    # prepare_data(supertest_sets, set_type='supertest', viz=False, cam_drop_frame=True)