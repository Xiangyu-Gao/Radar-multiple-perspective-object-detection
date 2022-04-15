import os
import numpy as np

from utils import get_sec
from config import camera_configs


def fix_cam_drop_frames(seq_path, label_names):
    ts_path = os.path.join(seq_path, camera_configs['time_stamp_name'])
    try:
        with open(ts_path) as ts_f:
            ts = ts_f.readlines()
    except:
        return label_names
    n_labels = len(ts)
    if int(float(ts[-1].rstrip()) * camera_configs['frame_rate']) == n_labels - 1:
        # no dropped frame
        return label_names
    label_names_new = [None] * n_labels
    for idx, line in enumerate(ts):
        time = float(line.rstrip())
        real_id = int(time * camera_configs['frame_rate'])
        if real_id < n_labels:
            label_names_new[real_id] = label_names[idx]
    # search for the nearest element with labels
    prev_nearest = - np.ones((n_labels, ), dtype=int)
    post_nearest = - np.ones((n_labels, ), dtype=int)
    prev_flag = -1
    post_flag = n_labels
    for idx, label in enumerate(label_names_new):
        if label is not None:
            prev_flag = idx
        else:
            prev_nearest[idx] = prev_flag
    for idx, label in reversed(list(enumerate(label_names_new))):
        if label is not None:
            post_flag = idx
        else:
            post_nearest[idx] = post_flag
    for idx in range(n_labels):
        if prev_nearest[idx] >= 0 and prev_nearest[idx] < n_labels and \
            post_nearest[idx] >= 0 and post_nearest[idx] < n_labels:
            if idx - prev_nearest[idx] <= post_nearest[idx] - idx:
                sup_idx = prev_nearest[idx]
            else:
                sup_idx = post_nearest[idx]
        elif not (prev_nearest[idx] >= 0 and prev_nearest[idx] < n_labels):
            sup_idx = post_nearest[idx]
        elif not (post_nearest[idx] >= 0 and post_nearest[idx] < n_labels):
            sup_idx = prev_nearest[idx]
        else:
            sup_idx = None
        if label_names_new[idx] is None:
            if sup_idx >= 0 and sup_idx < n_labels:
                # print('sup_idx:', sup_idx)
                # print('before:', label_names_new[idx])
                label_names_new[idx] = label_names_new[sup_idx]
                # print('after:', label_names_new[idx])
            else:
                raise ValueError
    return label_names_new


def calculate_frame_offset(start_time_txt):
    try:
        hhmmss = np.zeros((3, ))
        two_start_time = False
        with open(start_time_txt, 'r') as f:
            data = f.readlines()

        if len(data) <= 1:
            return 0, None, None

        start_time_readable = data[:3]
        if start_time_readable[2].rstrip() == '':
            two_start_time = True

        for idx, line in enumerate(start_time_readable[:2]):
            hhmmss_str = line.split(' ')[1]
            hhmmss[idx] = get_sec(hhmmss_str)

        offset01 = (hhmmss[1] - hhmmss[0]) * 30
        offset01 = int(round(offset01))

        if not two_start_time:
            offset02 = (hhmmss[2] - hhmmss[0]) * 30
            offset12 = (hhmmss[2] - hhmmss[1]) * 30

            offset02 = int(round(offset02))
            offset12 = int(round(offset12))

            return offset01, offset02, offset12

        return offset01, None, None

    except:
        return 0, None, None
