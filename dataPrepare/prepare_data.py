import os
import sys
import shutil
import numpy as np
import pickle
import argparse

from utils.confidence_map import generate_confmap, normalize_confmap, add_noise_channel
from utils.dataset_tools import fix_cam_drop_frames, calculate_frame_offset
from utils.read_annotations import read_3d_labels_txt, read_ra_labels_csv, read_3d_labels_refine_txt
from utils.visualization import visualize_confmap

from config import train_sets, test_sets, valid_sets, supertest_sets
from config import n_class, class_ids, camera_configs, radar_configs, rodnet_configs


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare RODNet data.')
    parser.add_argument('-m', '--mode', type=str, dest='mode', help='choose from train, valid, test, supertest')
    parser.add_argument('-dd', '--datadir', type=str, dest='data_dir', default='./data',
                        help='data directory to save the prepared data')
    args = parser.parse_args()
    return args


def prepare_data(sets, set_type='train', viz=False, cam_drop_frame=False):
    root_dir = sets['root_dir']
    root_dir_npy = '/mnt/sda/3DRadardata'
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
            print(seq_path)
            radar_mat_names = sorted(os.listdir(os.path.join(seq_path, rodnet_configs['data_folder'])))
            n_data = len(radar_mat_names)
            ra_frame_offset = calculate_frame_offset(os.path.join(seq_path, 'start_time.txt'))[0]
            start_id = int(float(radar_mat_names[0].split('.')[0].split('_')[-1]))
            if ra_frame_offset > 0 and cam_anno[dateid]:
                ra_frame_offset += 40
                n_data -= ra_frame_offset
                start_id = 40


            detail_list = [[], ra_frame_offset]
            confmap_list = [[], []]

            # create paths for data
            for fid in range(n_data):
                if radar_configs['data_type'] == 'RI' or radar_configs['data_type'] == 'AP':
                    path = os.path.join(root_dir, sets_dates[dateid], seq, chirp_folder_name, "%04d.npy")
                elif radar_configs['data_type'] == 'RISEP' or radar_configs['data_type'] == 'APSEP':
                    path = os.path.join(root_dir_npy, sets_dates[dateid], seq, chirp_folder_name, "%04d", "%06d.npy")
                else:
                    raise ValueError
                detail_list[0].append(path)

            if set_type == 'supertest':
                pickle.dump(detail_list, open(os.path.join(detail_dir, set_type, seq + '.pkl'), 'wb'))
                # no labels need to be saved
                continue
            else:
                if cam_anno[dateid]:
                    # use camera annotations
                    # TODO: add frame offset using function: calculate_frame_offset
                    label_names = sorted(os.listdir(os.path.join(seq_path, rodnet_configs['label_folder'])))
                    # 'dets_3d': 3d localization results from detector
                    # 'labels_dets_3d': 3d localization results from labeled bbox
                    n_labels = len(label_names)  # number of label files
                    if cam_drop_frame:
                        label_names = fix_cam_drop_frames(seq_path, label_names)
                    print(n_data, len(label_names))

                    for real_id in range(start_id, n_data + start_id):
                        sys.stdout.write('\r' + 'processing cam anno %s %04d/%04d' % (seq_path, real_id, n_data + start_id))
                        # for each frame
                        if 'refine' in rodnet_configs['label_folder']:
                            label_name = label_names[real_id-start_id]
                            obj_info = read_3d_labels_refine_txt(seq_path, label_name)
                        else:
                            label_name = label_names[real_id]
                            obj_info = read_3d_labels_txt(seq_path, label_name)

                        confmap_gt = np.zeros((n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                                              dtype=float)
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
                else:
                    # use labelled RAMap
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
                # save pkl files
                pickle.dump(confmap_list, open(os.path.join(confmap_dir, set_type, seq + '.pkl'), 'wb'))

            # save pkl files
            pickle.dump(detail_list, open(os.path.join(detail_dir, set_type, seq + '.pkl'), 'wb'))

            # end frames loop
        # end seqs loop
    # end dates loop


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
    detail_dir = os.path.join(data_dir, 'data_details')
    if not os.path.exists(confmap_dir):
        os.makedirs(confmap_dir)
    if not os.path.exists(detail_dir):
        os.makedirs(detail_dir)

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