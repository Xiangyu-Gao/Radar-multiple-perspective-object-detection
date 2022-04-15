import os
import json
import pandas as pd

from config import class_ids, data_sets


def csv2txt(date, seq_name):

    label_folder = os.path.join(data_root, date, seq_name, 'labels_dets')
    if not os.path.isdir(label_folder):
        os.makedirs(label_folder)

    image_names = sorted(os.listdir(os.path.join(data_root, date, seq_name, 'images_udst')))
    for img_name in image_names:
        img_idx = img_name.split('.')[0]
        label_name = img_idx + '.txt'
        with open(os.path.join(label_folder, label_name), 'w'):
            pass
    label_csv_name = os.path.join(data_root, date, seq_name, 'image_labels.csv')
    data = pd.read_csv(label_csv_name)
    n_row, n_col = data.shape

    for r in range(n_row):
        filename = data['filename'][r]
        img_idx = filename.split('.')[0]
        label_name = img_idx + '.txt'
        region_count = data['region_count'][r]
        region_id = data['region_id'][r]
        try:
            region_shape_attri = json.loads(data['region_shape_attributes'][r])
            region_attri = json.loads(data['region_attributes'][r])
        except Exception as e:
            print(e)
            print("Error in row:", r)
            print(data['region_shape_attributes'][r])
            print(data['region_attributes'][r])
            raise TypeError

        if region_count == 0:
            continue

        label_row = []
        class_name = region_attri['class'].rstrip().lower()
        try:
            class_id = class_ids[class_name]
        except:
            print('Wrong object label name in %s, %s' % (seq_name, filename))
            raise ValueError
        label_row.append(class_name)
        label_row.append(region_attri['truncation'])
        label_row.append(region_attri['occlusion'])
        label_row.append(region_attri['reachability'])
        label_row.append(str(region_shape_attri['x']))
        label_row.append(str(region_shape_attri['y']))
        label_row.append(str(region_shape_attri['x'] + region_shape_attri['width']))
        label_row.append(str(region_shape_attri['x'] + region_shape_attri['height']))
        for k in range(7):
            label_row.append('-1000')                                  # 3d dim & loc
        label_row.append('1.0')

        with open(os.path.join(label_folder, label_name), 'a+') as f:
            f.write(' '.join(label_row))
            if region_id < region_count - 1:
                f.write('\n')


if __name__ == '__main__':

    data_root = data_sets['root_dir']
    dates = data_sets['dates']
    for date in dates:
        seqs = sorted(os.listdir(os.path.join(data_root, date)))
        seqs = seqs
        for seq in seqs:
            print('Processing %s ...' % seq)
            csv2txt(date, seq)

