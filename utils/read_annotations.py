import os
import math
import pandas as pd
import json

from utils import find_nearest
from utils.mappings import confmap2ra, labelmap2ra

from config import class_ids
from config import radar_configs, rodnet_configs, t_cl2rh

range_grid = confmap2ra(radar_configs, name='range')
angle_grid = confmap2ra(radar_configs, name='angle')
range_grid_label = labelmap2ra(radar_configs, name='range')
angle_grid_label = labelmap2ra(radar_configs, name='angle')


def read_3d_labels_txt(seq_path, label_name, label_folder=rodnet_configs['label_folder']):

    label = open(os.path.join(seq_path, label_folder, label_name))
    obj_info = []

    for line in label:
        # for each object
        line = line.rstrip().split()
        type = line[0]
        try:
            class_id = class_ids[type]
        except:
            print('Wrong object label name in %s, %s' % (seq_path, label_name))
            raise ValueError
        # TODO: add filter to object class

        x = float(line[11]) - t_cl2rh[0]
        y = float(line[12]) - t_cl2rh[1]
        z = float(line[13]) - t_cl2rh[2]
        distance = math.sqrt(x ** 2 + z ** 2)
        angle = math.degrees(math.atan(x / z))
        if distance > rodnet_configs['rr_max'] or distance < rodnet_configs['rr_min']:
            # ignore the objects out of the range
            continue
        if angle > rodnet_configs['ra_max'] or angle < rodnet_configs['ra_min']:
            # ignore the objects out of the range
            continue
        angle = math.degrees(math.atan(x / z))  # in degree
        rng_idx, _ = find_nearest(range_grid, distance)
        agl_idx, _ = find_nearest(angle_grid, angle)
        obj_info.append([rng_idx, agl_idx, class_id])
        # print('%s: %s' % (label_name, type))

    return obj_info


def read_ra_labels_csv(seq_path):

    label_csv_name = os.path.join(seq_path, 'ramap_labels.csv')
    data = pd.read_csv(label_csv_name)
    n_row, n_col = data.shape
    obj_info_list = []
    cur_idx = -1

    for r in range(n_row):
        filename = data['filename'][r]
        frame_idx = int(filename.split('.')[0].split('_')[-1])
        if cur_idx == -1:
            obj_info = []
            cur_idx = frame_idx
        if frame_idx > cur_idx:
            obj_info_list.append(obj_info)
            obj_info = []
            cur_idx = frame_idx

        region_count = data['region_count'][r]
        region_id = data['region_id'][r]

        if region_count != 0:
            region_shape_attri = json.loads(data['region_shape_attributes'][r])
            region_attri = json.loads(data['region_attributes'][r])

            cx = region_shape_attri['cx']
            cy = region_shape_attri['cy']
            distance = range_grid_label[cy]
            angle = angle_grid_label[cx]
            if distance > rodnet_configs['rr_max'] or distance < rodnet_configs['rr_min']:
                continue
            if angle > rodnet_configs['ra_max'] or angle < rodnet_configs['ra_min']:
                continue
            rng_idx, _ = find_nearest(range_grid, distance)
            agl_idx, _ = find_nearest(angle_grid, angle)
            try:
                class_str = region_attri['class']
            except:
                print("missing class at row %d" % r)
                continue
            try:
                class_id = class_ids[class_str]
            except:
                if class_str == '':
                    print("no class label provided!")
                    raise ValueError
                else:
                    class_id = -1000
                    print("Warning class not found! %s %010d" % (seq_path, frame_idx))
            obj_info.append([rng_idx, agl_idx, class_id])

    obj_info_list.append(obj_info)

    return obj_info_list


def read_3d_labels_refine_txt(seq_path, label_name):

    label = open(os.path.join(seq_path, rodnet_configs['label_folder'], label_name))
    obj_info = []

    for line in label:
        # for each object
        line = line.rstrip().split()
        type = line[1]
        try:
            class_id = class_ids[type]
        except:
            print('Wrong object label name in %s, %s' % (seq_path, label_name))
            raise ValueError

        distance = float(line[3])
        angle = math.degrees(float(line[5]))
        if distance > rodnet_configs['rr_max'] or distance < rodnet_configs['rr_min']:
            # ignore the objects out of the range
            continue
        if angle > rodnet_configs['ra_max'] or angle < rodnet_configs['ra_min']:
            # ignore the objects out of the range
            continue
        obj_info.append([int(line[2]), int(line[4]), class_id])
        # print('%s: %s' % (label_name, type))

    return obj_info
