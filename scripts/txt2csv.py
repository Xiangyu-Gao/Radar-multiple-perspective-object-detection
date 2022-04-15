import os
import math
import json
import pandas as pd
import sys
sys.path.append('/home/admin-cmmb/Documents/RODNet_dop')

from utils import find_nearest
from utils.mappings import labelmap2ra

from config import class_ids, data_sets
from config import t_cl2rh

OLD_LABEL_IMAGES = True

if OLD_LABEL_IMAGES:
    radar_configs={
        'sample_freq': 4e6,
        'sweep_slope': 21.0017e12,
        'ramap_rsize_label': 122,
        'ramap_asize_label': 121,
        'crop_num': 3,
        'ra_min_label': -60,             # min radar angle
        'ra_max_label': 60,              # max radar angle
    }
else:
    from config import radar_configs


def convert(sets):
    """
    Convert 3d label txt to ramap csv for VIA
    :param sets: data_sets
    :return:
    """
    root_dir = sets['root_dir']
    sets_dates = sets['dates']
    seqs = ['2019_09_29_onrd000', '2019_09_29_onrd001', '2019_09_29_onrd003', '2019_09_29_onrd004',
            '2019_09_29_onrd017', '2019_09_29_onrd018']
    # seqs = ['2019_09_29_onrd000']

    range_grid = labelmap2ra(radar_configs, name='range')
    angle_grid = labelmap2ra(radar_configs, name='angle')

    for date in range(len(sets_dates)):
        # seqs = sorted(os.listdir(os.path.join(root_dir, sets_dates[date])))
        for seq in seqs:
            print("Converting label for sequence %s ..." % seq)
            try:
                seq_path = os.path.join(root_dir, sets_dates[date], seq)
                labels_path = os.path.join(seq_path, "dets_3d")
                images_path = os.path.join(seq_path, "WIN_RADAR_LABEL_IMAGE")
                file_attributes = '{}'
                region_shape_attributes = {"name": "point", "cx": 0, "cy": 0}
                region_attributes = {"class": None}
                columns = ['filename', 'file_size', 'file_attributes', 'region_count', 'region_id',
                           'region_shape_attributes', 'region_attributes']
                data = []
                images = sorted(os.listdir(images_path))
                labels = sorted(os.listdir(labels_path))
                for i in range(len(images)):
                    img_size = os.path.getsize(os.path.join(images_path, images[i]))
                    label = open(os.path.join(labels_path, labels[i]))
                    region_count = 0
                    obj_info = []
                    for line in label: # parse a label file
                        line = line.rstrip().split()
                        type = line[0]
                        try:
                            class_id = class_ids[type]
                        except:
                            print('Wrong object label name in %s, %s' % (seq_path, labels[i]))
                            raise ValueError
                        x = float(line[11]) - t_cl2rh[0]
                        y = float(line[12]) - t_cl2rh[1]
                        z = float(line[13]) - t_cl2rh[2]
                        distance = math.sqrt(x ** 2 + z ** 2)
                        angle = math.degrees(math.atan(x / z))  # in degree
                        rng_idx, _ = find_nearest(range_grid, distance)
                        agl_idx, _ = find_nearest(angle_grid, angle)
                        obj_info.append([rng_idx, agl_idx, type])
                        region_count += 1
                    for objid, obj in enumerate(obj_info):  # set up rows for different objs
                        row = []
                        row.append(images[i])
                        row.append(img_size)
                        row.append(file_attributes)
                        row.append(region_count)
                        row.append(objid)
                        if obj[1] <= 0 or obj[1] >= radar_configs['ramap_asize_label'] or \
                                obj[0] <= 0 or obj[0] >= radar_configs['ramap_rsize_label']:
                            continue
                        region_shape_attributes["cx"] = int(obj[1])
                        region_shape_attributes["cy"] = int(obj[0])
                        region_attributes["class"] = obj[2]
                        row.append(json.dumps(region_shape_attributes))
                        row.append(json.dumps(region_attributes))
                        data.append(row)
                df = pd.DataFrame(data, columns=columns)
                df.to_csv(os.path.join(seq_path, "ramap_labels_cvt.csv"), index=None, header=True)
                print("\tSuccess!")

            except Exception as e:
                print("\tLabel convert fails: %s" % e)


if __name__ == "__main__":
    convert(data_sets)
