import sys
sys.path.append("../")

import os
import shutil
import numpy as np

from config import data_sets
from config import n_class
from config import radar_configs, rodnet_configs
from utils.read_annotations import read_3d_labels_txt, read_ra_labels_csv
from utils.dataset_tools import fix_cam_drop_frames, calculate_frame_offset
from utils.visualization import visualize_anno_ramap


def viz_cam_anno(date, seq, cam_anno, viz=False):

    seq_path = os.path.join(data_root, date, seq)
    # radar_mat_names = sorted(os.listdir(os.path.join(seq_path, rodnet_configs['data_folder'])))
    # n_data = len(radar_mat_names)
    ra_frame_offset = calculate_frame_offset(os.path.join(seq_path, 'start_time.txt'))[0]
    # start_id = int(float(radar_mat_names[0].split('.')[0].split('_')[-1]))
    ramaps_anno_path = os.path.join(data_root, date, seq, 'ramaps_anno')
    if os.path.exists(ramaps_anno_path):
        shutil.rmtree(ramaps_anno_path)
    if not os.path.exists(ramaps_anno_path):
        os.makedirs(ramaps_anno_path)

    if cam_anno:
        # use camera annotations
        # TODO: add frame offset using function: calculate_frame_offset
        label_names = sorted(os.listdir(os.path.join(seq_path, 'dets_3d')))
        # 'dets_3d': 3d localization results from detector
        # 'labels_dets_3d': 3d localization results from labeled bbox
        n_labels = len(label_names)  # number of label files
        label_names = fix_cam_drop_frames(seq_path, label_names)

        for label_id in range(n_labels):
            rad_id = label_id + ra_frame_offset
            sys.stdout.write('\r' + 'processing cam anno %s %d/%d' % (seq_path, label_id+1, n_labels))
            # for each frame
            label_name = label_names[label_id]
            obj_info = read_3d_labels_txt(seq_path, label_name)
            path = os.path.join(data_root, date, seq, 'radar_chirps_win_RISEP', "0000", "%06d.npy" % rad_id)
            if not os.path.exists(path):
                continue
            ramap = np.load(path)
            figname = os.path.join(ramaps_anno_path, "%010d.jpg" % rad_id)
            visualize_anno_ramap(ramap, obj_info, figname, viz=viz)
        print()

    else:
        # use labelled RAMap
        try:
            obj_info_list = read_ra_labels_csv(seq_path)
        except Exception as e:
            print("Load sequence %s failed!" % seq_path)
            print(e)
            return

        for rad_id, obj_info in enumerate(obj_info_list):
            path = os.path.join(data_root, date, seq, 'radar_chirps_win_RISEP', "0000", "%06d.npy" % rad_id)
            ramap = np.load(path)
            figname = os.path.join(ramaps_anno_path, "%010d.jpg" % rad_id)
            visualize_anno_ramap(ramap, obj_info, figname, viz=viz)


if __name__ == '__main__':
    data_root = data_sets['root_dir']
    dates = data_sets['dates']
    cam_annos = data_sets['cam_anno']
    for date, cam_anno in zip(dates, cam_annos):
        seqs = sorted(os.listdir(os.path.join(data_root, date)))
        for seq in seqs:
            print(seq)
            viz_cam_anno(date, seq, cam_anno, viz=False)
