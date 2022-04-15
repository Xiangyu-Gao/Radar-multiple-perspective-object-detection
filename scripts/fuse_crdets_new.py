import os
import math
import numpy as np
import sys
sys.path.append("../")

from utils import find_nearest, detect_peaks
from utils.mappings import confmap2ra
from utils.confidence_map import normalize_confmap
from utils.visualization import visualize_confmap, visualize_confmaps_cr
from utils.visualization import visualize_fuse_crdets, visualize_fuse_crdets_compare
from utils.dataset_tools import calculate_frame_offset

from config import radar_configs, n_class, class_ids, class_table

ramap_rsize = radar_configs['ramap_rsize']
ramap_asize = radar_configs['ramap_asize']

range_grid = confmap2ra(radar_configs, name='range')
angle_grid = confmap2ra(radar_configs, name='angle', radordeg='rad')
agl_err = np.abs(angle_grid[:int(ramap_asize/2)] - angle_grid[1:int(ramap_asize/2)+1]) / 2
agl_err = np.concatenate((agl_err, np.flip(agl_err)))
# print(angle_grid)

TRUNCATE_ANGLE = 25.0

dist_dict = {
    'pedestrian': 0.5,
    'cyclist': 1.0,
    'car': 3.0,
}


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


def read_camera_dts(filename, offset=40):
    txts = sorted(os.listdir(filename))
    dts = []
    while len(dts) < len(txts) - offset:
        dts += [[]]
    for txt in txts:
        frameid = int(txt[:-4])
        if frameid < offset:
            continue
        frameid -= offset
        with open(os.path.join(filename, txt), 'r') as f:
            data = f.readlines()
        for obj_id, line in enumerate(data):
            # for each object
            line = line.rstrip().split()
            type = line[0]
            x1 = int(line[4])
            y1 = int(line[5])
            x2 = int(line[6])
            y2 = int(line[7])
            if (x2-x1)*(y2-y1) < 3000:
                continue
            x = float(line[11])
            y = float(line[12])
            z = float(line[13])
            distance = math.sqrt(x ** 2 + z ** 2)
            angle = math.atan(x / z)  # in radian
            rid, _ = find_nearest(range_grid, distance)
            aid, _ = find_nearest(angle_grid, angle)
            obj_dict = {
                'frame_id': frameid,
                'object_id': obj_id,
                'class': type,
                'range_id': rid,
                'angle_id': aid,
                'range': distance,
                'angle': angle,
            }
            dts[frameid].append(obj_dict)
    return dts


def read_radar_dts(filename, offset):
    dts = []
    cur_frame = 0
    obj_id = 0
    with open(filename, 'r') as f:
        data = f.readlines()
    for line in data:
        frameid, _, rid, aid, amp = line.rstrip().split()
        frameid = int(frameid) - 1
        if frameid < offset:
            continue
        frameid -= offset
        if cur_frame < frameid:
            obj_id = 0
            cur_frame = frameid
        rid = int(rid) - 4
        aid = int(aid) - 1
        amp = float(amp)
        if amp < 50:
            continue
        rng = range_grid[rid]
        agl = angle_grid[aid]
        obj_dict = {
            'frame_id': frameid,
            'object_id': obj_id,
            'range_id': rid,
            'angle_id': aid,
            'amplitude': amp,
            'range': rng,
            'angle': agl,
        }
        obj_id += 1

        # print(frameid)
        try:
            dts[frameid].append(obj_dict)
        except:
            while len(dts) <= frameid:
                dts += [[]]
            dts[frameid] = [obj_dict]

    return dts


def find_obj_loc(orgr, rngc, thetac, rngr, thetar):
    x0r, y0r = orgr
    a = math.tan(thetac) ** 2 + 1
    b = 2 * math.tan(thetac) * x0r + 2 * y0r
    c = x0r ** 2 + y0r ** 2 - rngr ** 2
    y = (- b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
    x = y * math.tan(thetac)
    # xc, yc = pol2cart(rngc, thetac)
    # xr, yr = pol2cart(rngr, thetar)
    # residualc = ((x - xc) ** 2 + (y - yc) ** 2) ** 0.5
    # residualr = ((x - xr) ** 2 + (y - yr) ** 2) ** 0.5
    rngt, thetat = cart2pol(x, y)
    res_rngc = rngc - rngt
    res_therar = thetar - thetat
    return (x, y), (rngt, thetat), (res_rngc, res_therar)


def calculate_dists(X, Y):
    sn,ss = X.shape
    tn,ss = Y.shape

    YT = Y.transpose()
    XYT = np.dot(X, YT)

    sqX = X * X
    sumsqX = np.sum(sqX, axis=1).reshape((sn, 1))
    sumsqXex = np.tile(sumsqX, (1, tn))
    sumsqXexT = sumsqXex.T

    sqedist = sumsqXex - 2 * XYT + sumsqXexT
    sqedist = np.around(sqedist, decimals=8)

    return np.sqrt(sqedist)


def get_groups(dists, rids, thres):
    nd0, nd1 = dists.shape
    edges = []
    set_dict = {}
    for id0 in range(nd0):
        for id1 in range(id0+1, nd1):
            dist = dists[id0, id1]
            if dist < thres:
                edges.append([id0, id1])
    for edge in edges:
        if rids[edge[0]] in set_dict:
            set_dict[rids[edge[0]]].append(rids[edge[1]])
        else:
            set_dict[rids[edge[0]]] = [rids[edge[1]]]

    point_sets = []
    seen_points = []
    for key, value in set_dict.items():
        if key not in seen_points:
            new_set = [key] + value
            seen_points.append(key)
            seen_points += value
            point_sets.append(new_set)

    return point_sets


def generate_camera_map(dtsc):
    map = np.zeros((ramap_rsize, ramap_asize, n_class))
    pps = []
    while len(pps) < n_class:
        pps += [[]]
    for dtid, dtc in enumerate(dtsc):
        rng_off = 0
        agl_off = 0
        map_local = np.zeros((ramap_rsize, ramap_asize, n_class))
        rid = dtc['range_id']
        aid = dtc['angle_id']
        rng = dtc['range']
        agl = dtc['angle']
        class_id = class_ids[dtc['class']]
        trun = False
        if abs(agl) > math.radians(TRUNCATE_ANGLE):
            trun = True
        sigma_r = 0.2 * rng + 1.0
        sigma_a = agl_err[aid] + math.radians(1.0)
        if trun:
            sigma_a += math.radians(5.0)
            agl_off = math.radians(1.0)
        try:
            pps[class_id].append((rid, aid))
        except:
            while len(pps) <= class_id:
                pps += [[]]
            pps[class_id] = [(rid, aid)]
        for r in range(ramap_rsize):
            for a in range(ramap_asize):
                center_r = rng + rng_off
                center_a = agl + np.sign(agl) * agl_off
                cur_r = range_grid[r]
                cur_a = angle_grid[a]
                off_r = cur_r - center_r
                off_a = cur_a - center_a
                pr = np.exp(-0.5 * (off_r / sigma_r) ** 2) / (sigma_r * (2 * np.pi) ** 0.5)
                pa = np.exp(-0.5 * (off_a / sigma_a) ** 2) / (sigma_a * (2 * np.pi) ** 0.5)
                value = pr * pa
                map_local[r, a, class_id] = value
        map_local[:, :, class_id] = normalize_confmap(map_local[:, :, class_id])
        map = np.maximum(map, map_local)
    return map, pps


def generate_radar_map(dtsr):
    map = np.zeros((ramap_rsize, ramap_asize))
    pps = []
    for dtid, dtr in enumerate(dtsr):
        map_local = np.zeros((ramap_rsize, ramap_asize))
        rid = dtr['range_id']
        aid = dtr['angle_id']
        rng = dtr['range']
        agl = dtr['angle']
        amp = dtr['amplitude']
        sigma_r = 0.1
        sigma_a = agl_err[aid] + math.radians(5.0)
        pps.append((rid, aid))
        for r in range(ramap_rsize):
            for a in range(ramap_asize):
                center_r = rng
                center_a = agl
                cur_r = range_grid[r]
                cur_a = angle_grid[a]
                off_r = cur_r - center_r
                off_a = cur_a - center_a
                pr = np.exp(-0.5 * (off_r / sigma_r) ** 2) / (sigma_r * (2 * np.pi) ** 0.5)
                pa = np.exp(-0.5 * (off_a / sigma_a) ** 2) / (sigma_a * (2 * np.pi) ** 0.5)
                value = pr * pa
                map_local[r, a] = value
        map = np.maximum(map, map_local)
    map = normalize_confmap(map)
    return map, pps


def grouping_on_maps(maps):

    dts_dicts = []
    for class_id in range(n_class):
        map_this = maps[class_id]
        peaks_row, peaks_col = detect_peaks(map_this)
        for rng_id, agl_id in zip(peaks_row, peaks_col):
            conf = map_this[rng_id, agl_id]
            if conf < 0.5:
                continue
            rng = range_grid[rng_id]
            agl = angle_grid[agl_id]
            peak_dict = {
                'class': class_table[class_id],
                'range': rng,
                'angle': agl,
                'range_id': rng_id,
                'angle_id': agl_id,
                'confidence': conf,
            }
            dts_dicts.append(peak_dict)
    dts_dicts = merge_peaks(dts_dicts)
    return dts_dicts


def merge_peaks(loc3d_dicts):
    xys_dict = {}
    ids_dict = {}
    del_ids = []
    for idx, loc3d_dict in enumerate(loc3d_dicts):
        cla = loc3d_dict['class']
        rng = loc3d_dict['range']
        agl = loc3d_dict['angle']
        x, y = pol2cart(rng, agl)
        try:
            ids_dict[cla].append(idx)
            xys_dict[cla].append([x, y])
        except:
            ids_dict[cla] = [idx]
            xys_dict[cla] = [[x, y]]
    for cla, xys in xys_dict.items():
        xys = np.array(xys)
        dists = calculate_dists(xys, xys)
        try:
            thres = dist_dict[cla]
        except:
            thres = 3.0
        sets = get_groups(dists, ids_dict[cla], thres=thres)
        for setone in sets:
            for idx, detid in enumerate(setone):
                if idx == 0:
                    continue
                del_ids.append(detid)
    del_ids = list(set(del_ids))
    if len(del_ids) > 0:
        for index in sorted(del_ids, reverse=True):
            del loc3d_dicts[index]
    return loc3d_dicts


def parse_points(dts_dicts):
    pps = []
    while len(pps) < n_class:
        pps += [[]]
    for dts_dict in dts_dicts:
        class_str = dts_dict['class']
        class_id = class_ids[class_str]
        rng_id = dts_dict['range_id']
        agl_id = dts_dict['angle_id']
        pps[class_id].append((rng_id, agl_id))
    return pps


if __name__ == '__main__':

    # data_root = '/mnt/nas_crdataset'
    data_root = '../data/tmp'
    date = '2019_09_29'
    # seq_name = '2019_09_29_onrd018'
    seq_names = sorted(os.listdir(os.path.join(data_root, date)))
    orgr = (0.11, 0.06)

    for seq_name in seq_names:

        cr_offset, _, _ = calculate_frame_offset(os.path.join(data_root, date, seq_name, 'start_time.txt'))
        dtsc = read_camera_dts(os.path.join(data_root, date, seq_name, 'dets_3d'), offset=40)
        # dtsc = read_camera_dts(os.path.join(data_root, date, seq_name, 'dets_3d'), offset=0)
        dtsr = read_radar_dts(os.path.join(data_root, date, seq_name, 'cfar_dets.txt'), offset=40+cr_offset)
        # dtsr = read_radar_dts(os.path.join(data_root, date, seq_name, 'cfar_dets.txt'), offset=0)
        seq_length = len(dtsr)
        dtsc = dtsc[:seq_length]

        if not os.path.exists(os.path.join(data_root, date, seq_name, 'dets_refine')):
            os.makedirs(os.path.join(data_root, date, seq_name, 'dets_refine'))
        if not os.path.exists(os.path.join(data_root, date, seq_name, 'dets_refine_prob')):
            os.makedirs(os.path.join(data_root, date, seq_name, 'dets_refine_prob'))
        if not os.path.exists(os.path.join(data_root, date, seq_name, 'dets_refine_viz')):
            os.makedirs(os.path.join(data_root, date, seq_name, 'dets_refine_viz'))

        for frameid, (dtsc_in_frame, dtsr_in_frame) in enumerate(zip(dtsc, dtsr)):

            print(seq_name, frameid)
            # if frameid < 107:
            #     continue

            mapc, ppsc = generate_camera_map(dtsc_in_frame)
            mapc = np.transpose(mapc, axes=(2, 0, 1))
            # visualize_confmap(mapc, pps)
            mapr, ppsr = generate_radar_map(dtsr_in_frame)
            # visualize_confmap(mapr, pps)

            mapcrs = []
            for class_id in range(n_class):
                mapc_this = mapc[class_id]
                mapcr = np.multiply(mapc_this, mapr)
                print(np.max(mapc_this), np.max(mapr), np.max(mapcr))
                mapcrs.append(mapcr)
            mapcrs = np.array(mapcrs)
            dts_dicts = grouping_on_maps(mapcrs)
            ppres = parse_points(dts_dicts)
            figname = os.path.join(data_root, date, seq_name, 'dets_refine_prob/%06d.jpg' % frameid)
            visualize_confmaps_cr(mapc, mapr, mapcrs, ppsc=ppsc, ppsr=ppsr, ppres=ppres, figname=figname)

            # visualize
            figname = os.path.join(data_root, date, seq_name, 'dets_refine_viz/%06d.jpg' % frameid)
            img_path = os.path.join(data_root, date, seq_name, 'images_hist_0', '%010d.jpg' % (frameid + 40))
            # chirp_data = np.load(os.path.join(data_root, date, seq_name, 'radar_chirps_win_RISEP/0000/%06d.npy' % (frameid + 40+cr_offset)))
            chirp_data = np.load(os.path.join(data_root, date, seq_name, 'radar_chirps_win_RISEP/0000/%06d.npy' % (frameid)))
            visualize_fuse_crdets_compare(img_path, chirp_data, dtsc_in_frame, dtsr_in_frame, dts_dicts, figname=figname)

            out_path = os.path.join(data_root, date, seq_name, 'dets_refine', '%010d.txt' % frameid)
            with open(out_path, 'w') as f:
                for loc3d_dict in dts_dicts:
                    f.write("%s %d %.4f %d %.4f %.4f\n" % (loc3d_dict['class'],
                                                              loc3d_dict['range_id'], loc3d_dict['range'],
                                                              loc3d_dict['angle_id'], loc3d_dict['angle'],
                                                              loc3d_dict['confidence']))
