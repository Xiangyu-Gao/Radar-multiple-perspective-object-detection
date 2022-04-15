import os
import math
import sys
import numpy as np

from utils import find_nearest
from utils.mappings import confmap2ra
from utils.visualization import visualize_fuse_crdets, visualize_fuse_crdets_compare
from utils.dataset_tools import calculate_frame_offset

from config import radar_configs, n_class, class_ids, class_table


fft_Ang = 128
w = np.linspace(-1, 1, fft_Ang)         # angle_grid
angle_grid = np.arcsin(w)                 # [-1,1]->[-pi/2,pi/2]
agl_err = np.abs(angle_grid[:int(fft_Ang/2)] - angle_grid[1:int(fft_Ang/2)+1]) / 2
agl_err = np.concatenate((agl_err, np.flip(agl_err)))

range_grid = confmap2ra(radar_configs, name='range')
TRUNCATE_ANGLE = 20.0

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


def dts_match_in_frame(dtsc, dtsr, orgr=(0.11, 0.06)):
    ndtsc = len(dtsc)
    ndtsr = len(dtsr)
    loc3d_fuse = np.zeros((ndtsc, ndtsr, 2))  # [x, y]
    residuals = np.zeros((ndtsc, ndtsr, 2))   # [camera range error, radar angle error]
    matches = np.zeros((ndtsc, ndtsr))
    for cid, dtc in enumerate(dtsc):
        # determine if truncated in camera
        trun = False
        if abs(dtc['angle']) > math.radians(TRUNCATE_ANGLE):
            trun = True
        for rid, dtr in enumerate(dtsr):
            (x, y), (rng, agl), (res_rngc, res_therar) = \
                find_obj_loc(orgr, dtc['range'], dtc['angle'], dtr['range'], dtr['angle'])
            loc3d_fuse[cid, rid, 0] = rng
            loc3d_fuse[cid, rid, 1] = agl
            residuals[cid, rid, 0] = res_rngc
            residuals[cid, rid, 1] = res_therar
            rng_thres = 0.2 * dtr['range'] + 1.0
            agl_id, _ = find_nearest(angle_grid, dtr['angle'])
            agl_thres = agl_err[agl_id] + math.radians(10.0)
            if trun:  # if truncated in camera, increase angle threshold
                if np.abs(res_rngc) < rng_thres and \
                        (agl >= 0 and - agl_thres < res_therar < agl_thres + math.radians(10.0) or
                                agl < 0 and - agl_thres - math.radians(10.0) < res_therar < agl_thres):
                    matches[cid, rid] = 1.0
            else:
                if np.abs(res_rngc) < rng_thres and np.abs(res_therar) < agl_thres:
                    matches[cid, rid] = 1.0
    return loc3d_fuse, matches, residuals


def remove_matches(frameid, dtsc, dtsr, loc3d, matches, residuals):

    loc3d_dicts = []

    # grouping nearby radar dets matched with same camera det
    matchesc = np.sum(matches, axis=1)
    cids = np.where(matchesc > 1.0)[0]
    n_matches = len(cids)
    if n_matches != 0:
        for cid in cids:
            # only one matched camera det for each radar det
            this_match = matches[cid, :]
            rids = np.where(this_match == 1.0)[0]
            ptsr = np.zeros((len(rids), 2))
            for idx, rid in enumerate(rids):
                rng = dtsr[rid]['range']
                agl = dtsr[rid]['angle']
                x, y = pol2cart(rng, agl)
                ptsr[idx, 0] = x
                ptsr[idx, 1] = y
            dists = calculate_dists(ptsr, ptsr)
            try:
                thres = dist_dict[dtsc[cid]['class']]
            except:
                thres = 3.0
            sets = get_groups(dists, rids, thres=thres)
            for set in sets:
                xs = []
                ys = []
                for rid in set:
                    rng = dtsr[rid]['range']
                    agl = dtsr[rid]['angle']
                    x, y = pol2cart(rng, agl)
                    xs.append(x)
                    ys.append(y)
                x_avg = sum(xs) / float(len(xs))
                y_avg = sum(ys) / float(len(ys))
                rng_avg, agl_avg = cart2pol(x_avg, y_avg)
                # rng_avg_id, agl_avg_id = ra2idx(rng_avg, agl_avg)
                for rid in set:
                    # dtsr[rid]['range_id'] = rng_avg_id
                    # dtsr[rid]['angle_id'] = agl_avg_id
                    # dtsr[rid]['range'] = rng_avg
                    # dtsr[rid]['angle'] = agl_avg
                    loc3d[cid, rid, 0] = rng_avg
                    loc3d[cid, rid, 1] = agl_avg

    # one radar det matched with multiple camera dets
    matchesr = np.sum(matches, axis=0)
    rids = np.where(matchesr > 1.0)[0]
    n_matches = len(rids)
    if n_matches != 0:
        for rid in rids:
            # only one matched camera det for each radar det
            this_match = matches[:, rid]
            cids = np.where(this_match == 1.0)[0]
            res_min = np.pi
            cid_select = -1
            for cid in cids:  # find min angle distance btw dtc and dtr
                agl_res = np.abs(residuals[cid, rid, 1])
                if agl_res < res_min:
                    res_min = agl_res
                    cid_select = cid
            assert cid_select != -1
            for cid in cids:
                if cid != cid_select:
                    matches[cid, rid] = 0.0

    # one camera det to multi radar dets
    for cid, dtc in enumerate(dtsc):
        # determine if truncated in camera
        trun = False
        if abs(dtc['angle']) > math.radians(TRUNCATE_ANGLE):
            trun = True
        this_matches = matches[cid, :]
        rids = np.where(this_matches == 1)[0]
        n_matches = len(rids)
        if n_matches == 0:  # no matches
            continue
        elif n_matches == 1:
            rid = rids[0]
            rng = loc3d[cid, rid, 0]
            agl = loc3d[cid, rid, 1]
            rng_id, _ = find_nearest(range_grid, rng)
            agl_id, _ = find_nearest(angle_grid, agl)
            obj_dict = {
                'frame_id': frameid,
                'object_id': dtc['object_id'],
                'class': dtc['class'],
                'range_id': rng_id,
                'angle_id': agl_id,
                'range': rng,
                'angle': agl,
                'amplitude': dtsr[rid]['amplitude']
            }
            loc3d_dicts.append(obj_dict)
        else:  # find best radar detection(s)
            rid_select = -1
            if trun:  # if truncated, consider distance
                res_min = 10000.0
                for rid in rids:  # find min angle distance btw dtc and dtr
                    dx, dy = pol2cart(residuals[cid, rid, 0], residuals[cid, rid, 1])
                    dist_res = dx ** 2 + dy ** 2
                    if dist_res < res_min:
                        res_min = dist_res
                        rid_select = rid
            else:  # if not truncated, consider angle only
                res_min = np.pi
                for rid in rids:  # find min angle distance btw dtc and dtr
                    agl_res = np.abs(residuals[cid, rid, 1])
                    if agl_res < res_min:
                        res_min = agl_res
                        rid_select = rid
            assert rid_select != -1
            rid = rid_select
            if trun:
                rng = dtsr[rid]['range']
                agl = dtsr[rid]['angle']
                rng_id = dtsr[rid]['range_id']
                agl_id = dtsr[rid]['angle_id']
            else:
                rng = loc3d[cid, rid, 0]
                agl = loc3d[cid, rid, 1]
                rng_id, _ = find_nearest(range_grid, rng)
                agl_id, _ = find_nearest(angle_grid, agl)
            obj_dict = {
                'frame_id': frameid,
                'object_id': dtc['object_id'],
                'class': dtc['class'],
                'range_id': rng_id,
                'angle_id': agl_id,
                'range': rng,
                'angle': agl,
                'amplitude': dtsr[rid]['amplitude']
            }
            loc3d_dicts.append(obj_dict)

    return loc3d_dicts


def grouping_dets(loc3d_dicts):
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


if __name__ == '__main__':

    data_root = '/mnt/nas_crdataset'
    date = '2019_09_29'
    # seq_names = ['2019_09_29_onrd003']
    seq_names = sorted(os.listdir(os.path.join(data_root, date)))
    orgr = (0.11, 0.06)

    for seq_name in seq_names:
        cr_offset, _, _ = calculate_frame_offset(os.path.join(data_root, date, seq_name, 'start_time.txt'))
        try:
            dtsc = read_camera_dts(os.path.join(data_root, date, seq_name, 'dets_3d'), offset=40)
            dtsr = read_radar_dts(os.path.join(data_root, date, seq_name, 'cfar_dets.txt'), offset=40+cr_offset)
        except:
            print("error! ignore this sequence", seq_name)
            continue
        seq_length = len(dtsr)
        dtsc = dtsc[:seq_length]
        if not os.path.exists(os.path.join(data_root, date, seq_name, 'dets_refine')):
            os.makedirs(os.path.join(data_root, date, seq_name, 'dets_refine'))
        n_refine = len(os.listdir(os.path.join(data_root, date, seq_name, 'dets_refine')))
        if n_refine == seq_length:
            print(seq_name, 'no need to rerun')
            continue

        # loc3d_all = []
        for frameid, (dtsc_in_frame, dtsr_in_frame) in enumerate(zip(dtsc, dtsr)):
            sys.stdout.write('\r%s/%d' % (seq_name, frameid))
            # if frameid < 107:
            #     continue
            loc3d_fuse, matches, residuals = dts_match_in_frame(dtsc_in_frame, dtsr_in_frame)
            loc3d_dicts = remove_matches(frameid, dtsc_in_frame, dtsr_in_frame, loc3d_fuse, matches, residuals)
            loc3d_dicts = grouping_dets(loc3d_dicts)

            chirp_data = np.load(os.path.join(data_root, date, seq_name, 'radar_chirps_win_RISEP/0000/%06d.npy' % (frameid + 40+cr_offset)))
            # visualize_fuse_crdets(chirp_data, dtsc_in_frame, figname='../data/tmp/dets_viz/%06d_dtsc.jpg'%frameid)
            # visualize_fuse_crdets(chirp_data, dtsr_in_frame, figname='../data/tmp/dets_viz/%06d_dtsr.jpg'%frameid)
            # visualize_fuse_crdets(chirp_data, loc3d_dicts, figname='../data/tmp/dets_viz/%06d_dts_final.jpg'%frameid)
            if not os.path.exists(os.path.join(data_root, date, seq_name, 'dets_refine')):
                os.makedirs(os.path.join(data_root, date, seq_name, 'dets_refine'))
            if not os.path.exists(os.path.join(data_root, date, seq_name, 'dets_refine_viz')):
                os.makedirs(os.path.join(data_root, date, seq_name, 'dets_refine_viz'))
            figname = os.path.join(data_root, date, seq_name, 'dets_refine_viz/%06d.jpg' % frameid)
            img_path = os.path.join(data_root, date, seq_name, 'images_hist_0', '%010d.jpg' % (frameid + 40))
            visualize_fuse_crdets_compare(img_path, chirp_data, dtsc_in_frame, dtsr_in_frame, loc3d_dicts, figname=figname)
            # loc3d_all.append(loc3d_dicts)

            out_path = os.path.join(data_root, date, seq_name, 'dets_refine', '%010d.txt' % frameid)
            with open(out_path, 'w') as f:
                for loc3d_dict in loc3d_dicts:
                    f.write("%d %s %d %.4f %d %.4f %.4f\n" % (loc3d_dict['object_id'], loc3d_dict['class'],
                                                              loc3d_dict['range_id'], loc3d_dict['range'],
                                                              loc3d_dict['angle_id'], loc3d_dict['angle'],
                                                              loc3d_dict['amplitude']))
