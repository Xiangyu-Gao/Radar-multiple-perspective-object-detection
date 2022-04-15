import os
import math
import numpy as np

from utils import find_nearest
from utils.mappings import confmap2ra

from config import radar_configs

range_grid = confmap2ra(radar_configs, name='range')
angle_grid = confmap2ra(radar_configs, name='angle', radordeg='rad')


def pol2cart(rho, phi):
    x = rho * np.sin(phi)
    y = rho * np.cos(phi)
    return x, y


def cart2pol(x, y):
    rho = (x * x + y * y) ** 0.5
    phi = np.arctan2(x, y)
    return rho, phi


def ra2idx(rng, agl):
    rng_id, _ = find_nearest(range_grid, rng)
    agl_id, _ = find_nearest(angle_grid, agl)
    return rng_id, agl_id


def generate_bbox_ramap(rng, agl, bbox_size):
    x, y = pol2cart(rng, agl)
    x1 = x - bbox_size / 2
    x2 = x + bbox_size / 2
    y1 = y - bbox_size / 2
    y2 = y + bbox_size / 2
    r1, a1 = cart2pol(x1, y1)
    r2, a2 = cart2pol(x1, y2)
    r3, a3 = cart2pol(x2, y1)
    r4, a4 = cart2pol(x2, y2)
    rmin = min(r1, r3)
    rmax = max(r2, r4)
    amin = min(a1, a2)
    amax = max(a3, a4)
    if rmin < radar_configs['rr_min']:
        rmin = radar_configs['rr_min']
    if rmax > radar_configs['rr_max']:
        rmax = radar_configs['rr_max']
    if amin < math.radians(radar_configs['ra_min']):
        amin = math.radians(radar_configs['ra_min'])
    if amax > math.radians(radar_configs['ra_max']):
        amax = math.radians(radar_configs['ra_max'])

    rmin_id, amin_id = ra2idx(rmin, amin)
    rmax_id, amax_id = ra2idx(rmax, amax)
    return rmin_id, amin_id, rmax_id, amax_id


data_root = '/mnt/nas_crdataset'
dates = ['2019_09_29']
bbox_sizes_dict = {
    'pedestrian': 1.0,
    'cyclist': 2.0,
    'car': 4.0,
    'truck': 2.0,
    'train': 6.0,
}

for date in dates:
    seqs = sorted(os.listdir(os.path.join(data_root, date)))
    for seq in seqs:
        print(seq)
        seq_path = os.path.join(data_root, date, seq)
        dets_path = os.path.join(seq_path, 'dets_refine')
        dets_path_new = os.path.join(seq_path, 'dets_refine_bbox')
        if os.path.exists(dets_path):
            txts = sorted(os.listdir(dets_path))
            if not os.path.exists(dets_path_new):
                os.makedirs(dets_path_new)
            for txt in txts:
                txt_path = os.path.join(dets_path, txt)
                txt_path_new = os.path.join(dets_path_new, txt)
                with open(txt_path, 'r') as f:
                    data = f.readlines()
                with open(txt_path_new, 'w') as f:
                    for line in data:
                        object_id, class_str, rng_id, rng, agl_id, agl, amp = line.rstrip().split()
                        bbox_size = bbox_sizes_dict[class_str]
                        rid1, aid1, rid2, aid2 = generate_bbox_ramap(float(rng), float(agl), bbox_size)
                        f.write("%s %s %s %s %s %s %d %d %d %d\n" % (class_str, rng_id, rng, agl_id, agl, amp,
                                                                     rid1, aid1, rid2, aid2))
