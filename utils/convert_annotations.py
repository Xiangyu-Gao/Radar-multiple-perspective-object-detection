import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import math
import json
import pandas as pd

from utils import find_nearest
from utils.mappings import labelmap2ra
from utils.read_annotations import read_ra_labels_csv
from config import radar_configs

release_dataset_label_map = {0: 'person',
                             2: 'car',
                             3: 'motorbike',
                             5: 'bus',
                             7: 'truck',
                             80: 'cyclist'}


def convert(seq_path):
    range_grid = labelmap2ra(radar_configs, name='range')
    angle_grid = labelmap2ra(radar_configs, name='angle')

    images_path = os.path.join(seq_path, "images_0")
    label_path = os.path.join(seq_path, "text_labels")
    images = sorted(os.listdir(images_path))
    files = sorted(os.listdir(label_path))

    file_attributes = '{}'
    region_shape_attributes = {"name": "point", "cx": 0, "cy": 0}
    region_attributes = {"class": None}
    columns = ['filename', 'file_size', 'file_attributes', 'region_count', 'region_id',
               'region_shape_attributes', 'region_attributes']
    data = []

    for file in files:
        file_dir = os.path.join(label_path, file)
        label = open(file_dir)
        img_name = file.replace("csv", "jpg")
        img_size = os.path.getsize(os.path.join(images_path, img_name))
        region_count = 0
        obj_info = []

        # parse a label file
        for line in label:
            line = line.rstrip().split(',')
            type = release_dataset_label_map[int(line[1])]

            x = int(float(line[2]))
            y = int(float(line[3]))
            distance = math.sqrt(x ** 2 + y ** 2)
            angle = math.degrees(math.atan(x / y))  # in degree
            rng_idx, _ = find_nearest(range_grid, distance)
            agl_idx, _ = find_nearest(angle_grid, angle)
            obj_info.append([rng_idx, agl_idx, type])
            region_count += 1

        for objId, obj in enumerate(obj_info):  # set up rows for different objs
            row = []
            row.append(img_name)
            row.append(img_size)
            row.append(file_attributes)
            row.append(region_count)
            row.append(objId)
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
    df.to_csv(os.path.join(seq_path, "ramap_labels.csv"), index=None, header=True)
    print("\tSuccess!")

    return


if __name__ == "__main__":
    path = 'F:\\data\Automotive\\2019_04_09_cms1000'
    convert(path)
    res = read_ra_labels_csv(path)
    print(len(res))

