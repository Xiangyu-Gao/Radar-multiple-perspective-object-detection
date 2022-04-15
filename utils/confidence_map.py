import numpy as np
import math

from utils import cart2pol_ramap, pol2cart_ramap, find_nearest
from utils import rotate, dist_point_segment, rotate_conf_pattern, is_inside_triangle
from utils.mappings import confmap2ra
from utils.visualization import visualize_confmap

from config import n_class, class_table, radar_configs, confmap_sigmas, confmap_length, confmap_sigmas_interval, object_sizes

# MAX_SIGMA = 30
# MIN_SIGMA = 8

range_grid = confmap2ra(radar_configs, name='range')
angle_grid = confmap2ra(radar_configs, name='angle')


def generate_confmap(obj_info, gaussian_thres=36, type='ra'):
    """
    Generate confidence map for given range/angle indices, class_id, and class_sigma
    :param rng_idx:
    :param agl_idx:
    :param class_id: object class id
    :param sigma: std for gaussians
    :param gaussian_thres: threshold for gaussian distribution in confmaps
    :return:
    """
    if type == 'ra':
        confmap = np.zeros((n_class, radar_configs['ramap_rsize'], radar_configs['ramap_asize']), dtype=float)
        for rng_idx, agl_idx, class_id in obj_info:
            if class_id < 0:
                continue
            try:
                class_name = class_table[class_id]
            except:
                continue
            sigma = 2 * np.arctan(confmap_length[class_table[class_id]] / (2 * range_grid[rng_idx])) * \
                    confmap_sigmas[class_table[class_id]]
            sigma_interval = confmap_sigmas_interval[class_table[class_id]]
            if sigma > sigma_interval[1]:
                sigma = sigma_interval[1]
            if sigma < sigma_interval[0]:
                sigma = sigma_interval[0]
            for i in range(radar_configs['ramap_rsize']):
                for j in range(radar_configs['ramap_asize']):
                    distant = (((rng_idx - i) * 2) ** 2 + (agl_idx - j) ** 2) / sigma**2
                    if distant < gaussian_thres:  # threshold for confidence maps
                        value = np.exp(- distant / 2) / (2 * math.pi)
                        confmap[class_id, i, j] = value if value > confmap[class_id, i, j] else confmap[class_id, i, j]
    else:
        confmap = np.zeros((n_class, radar_configs['ramap_rsize'], radar_configs['ramap_vsize']), dtype=float)
        for rng_idx, agl_idx, dop_idx, class_id in obj_info:
            if class_id < 0:
                continue
            try:
                class_name = class_table[class_id]
            except:
                continue
            sigma = 2 * np.arctan(confmap_length[class_table[class_id]] / (2 * range_grid[rng_idx])) * \
                    confmap_sigmas[class_table[class_id]]
            sigma_interval = confmap_sigmas_interval[class_table[class_id]]
            if sigma > sigma_interval[1]:
                sigma = sigma_interval[1]
            if sigma < sigma_interval[0]:
                sigma = sigma_interval[0]
            if type == 'rv':
                for i in range(radar_configs['ramap_rsize']):
                    for j in range(radar_configs['ramap_vsize']):
                        distant = (((rng_idx - i) * 2) ** 2 + ((dop_idx - j) * 2) ** 2) / sigma ** 2
                        if distant < gaussian_thres:  # threshold for confidence maps
                            value = np.exp(- distant / 2) / (2 * math.pi)
                            confmap[class_id, i, j] = value if value > confmap[class_id, i, j] else confmap[class_id, i, j]
            elif type == 'av':
                for i in range(radar_configs['ramap_asize']):
                    for j in range(radar_configs['ramap_vsize']):
                        distant = ((agl_idx - i) ** 2 + ((dop_idx - j) * 2) ** 2 ) / sigma ** 2
                        if distant < gaussian_thres:  # threshold for confidence maps
                            value = np.exp(- distant / 2) / (2 * math.pi)
                            confmap[class_id, i, j] = value if value > confmap[class_id, i, j] else confmap[class_id, i, j]


    return confmap


def normalize_confmap(confmap):
    conf_min = np.min(confmap)
    conf_max = np.max(confmap)
    if conf_max - conf_min != 0:
        confmap_norm = (confmap - conf_min) / (conf_max - conf_min)
    else:
        confmap_norm = confmap
    return confmap_norm


def add_noise_channel(confmap):
    confmap_new = np.zeros((n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']), dtype=float)
    confmap_new[:n_class, :, :] = confmap
    conf_max = np.max(confmap)
    confmap_new[n_class, :, :] = 1.0 - conf_max
    return confmap_new


def vehicle_confmap(raloc, width, length, ori, gaussian_thres=10):
    """
    :param raloc    Range-Angle location (meters/degrees)
    :param width    width of the vehicle
    :param length   length of the vehicle
    :param ori      orientation of the vehicle (degrees)
    """
    rng_idx, agl_idx = raloc
    rng = range_grid[rng_idx]
    angle = angle_grid[agl_idx]
    x = rng * math.sin(math.radians(angle))
    y = rng * math.cos(math.radians(angle))
    ori_rad = math.radians(ori)

    p1 = (x - width/2, y + length/2)
    p2 = (x + width/2, y + length/2)
    p3 = (x + width/2, y - length/2)
    p4 = (x - width/2, y - length/2)
    p1 = rotate(origin=(x, y), point=p1, angle=ori_rad)
    p2 = rotate(origin=(x, y), point=p2, angle=ori_rad)
    p3 = rotate(origin=(x, y), point=p3, angle=ori_rad)
    p4 = rotate(origin=(x, y), point=p4, angle=ori_rad)
    ps = [p1, p2, p3, p4]
    dists = [math.sqrt(p[0] ** 2 + p[1] ** 2) for p in ps]
    agls = [math.atan2(p[1], p[0]) for p in ps]
    dists_sortids = sorted(range(4), key=lambda k: dists[k])
    agls_sortids = sorted(range(4), key=lambda k: agls[k])

    vis_flag = [False, False, False, False]
    vis_flag[dists_sortids[0]] = True
    vis_flag[agls_sortids[0]] = True
    vis_flag[agls_sortids[-1]] = True
    n_vis = sum(vis_flag)
    ps_vis = []

    segments = []
    for pid in range(4):
        if vis_flag[pid]:
            ps_vis.append(ps[pid])
        pid2 = pid + 1
        if pid2 == 4:
            pid2 = 0
        if vis_flag[pid] and vis_flag[pid2]:
            segments.append((ps[pid], ps[pid2]))
    if vis_flag[0] and vis_flag[2]:
        segments.append((ps[0], ps[2]))
    if vis_flag[1] and vis_flag[3]:
        segments.append((ps[1], ps[3]))

    n_segs = len(segments)
    confmap = np.zeros((radar_configs['ramap_rsize'], radar_configs['ramap_asize']), dtype=float)
    pps = []

    for segid in range(n_segs):
        for ri in range(radar_configs['ramap_rsize']):
            for aj in range(radar_configs['ramap_asize']):
                rng_cur = range_grid[ri]
                agl_cur = angle_grid[aj]

                x_cur, y_cur = pol2cart_ramap(rng_cur, math.radians(agl_cur))

                sigma = 2 * np.arctan(confmap_length['car'] / (2 * rng_cur)) * confmap_sigmas['car'] / 20
                dist, (projx, projy) = dist_point_segment(point=(x_cur, y_cur), segment=segments[segid])

                projr, projtheta = cart2pol_ramap(projx, projy)
                projtheta = math.degrees(projtheta)
                rngid_proj, _ = find_nearest(range_grid, projr)
                aglid_proj, _ = find_nearest(angle_grid, projtheta)
                # pps.append((projtheta, projr))

                dr = rngid_proj - ri
                dtheta = aglid_proj - aj
                distant = (dtheta * dtheta / 16 + dr * dr)
                if n_vis == 2:
                    if is_inside_triangle(ps_vis[0], ps_vis[1], (x, y), (x_cur, y_cur)):
                        distant = 0
                else:  # n_vis == 3
                    if is_inside_triangle(ps_vis[0], ps_vis[1], ps_vis[2], (x_cur, y_cur)):
                        distant = 0
                        # pps.append((ri, aj))
                if distant < gaussian_thres:  # threshold for confidence maps
                    value = np.exp(- distant / 2 / (rng_cur * sigma**2))
                    confmap[ri, aj] = value if value > confmap[ri, aj] else confmap[ri, aj]

    for ri in range(radar_configs['ramap_rsize']):
        for aj in range(radar_configs['ramap_asize']):
            rng_cur = range_grid[ri]
            agl_cur = angle_grid[aj]
            x_cur = rng_cur * math.sin(math.radians(agl_cur))
            y_cur = rng_cur * math.cos(math.radians(agl_cur))
            sigma = 2 * np.arctan(confmap_length['car'] / (2 * rng_cur)) * confmap_sigmas['car'] / 20
            dx = x - x_cur
            dy = y - y_cur
            dx, dy = rotate_conf_pattern(dx, dy, ori_rad)
            distant = (dx * dx + dy * dy / 2)
            if distant < gaussian_thres:  # threshold for confidence maps
                value = np.exp(- distant / 2 / (rng_cur * sigma**2)) * 1.2
                confmap[ri, aj] = value if value > confmap[ri, aj] else confmap[ri, aj]

    visualize_confmap(confmap, pps)


if __name__ == '__main__':
    vehicle_confmap(raloc=(50, 90), width=1.5, length=3.5, ori=60)
