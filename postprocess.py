import numpy as np

from utils import detect_peaks
from utils.mappings import confmap2ra
from utils.ols import get_ols_btw_objects
from utils.visualization import visualize_postprocessing

from config import class_ids, class_table, n_class
from config import rodnet_configs, radar_configs


def search_surround(peak_conf, row, col, conf_valu, search_size):
    
    height = peak_conf.shape[0]
    width = peak_conf.shape[1]
    half_size = int((search_size - 1) / 2)
    row_start = max(half_size, row - half_size)
    row_end = min(height - half_size - 1, row + half_size)
    col_start = max(half_size, col - half_size)
    col_end = min(width - half_size - 1, col + half_size)
    # print(row_start)
    No_bigger = True
    for i in range(row_start, row_end + 1):
        for j in range(col_start, col_end + 1):
            if peak_conf[i, j] > conf_valu:
                # current conf is not big enough, skip this peak
                No_bigger = False
                break

    return No_bigger, [row_start, row_end, col_start, col_end]


def peak_mapping(peak_conf, peak_class, list_row, list_col, confmap, search_size, o_class):

    for i in range(len(list_col)):
        row_id = list_row[i]
        col_id = list_col[i]
        conf_valu = confmap[row_id, col_id]
        
        flag, indices = search_surround(peak_conf, row_id, col_id, conf_valu, search_size)
        if flag:
            # clear all detections in search window
            search_width = indices[1] - indices[0] + 1
            search_height = indices[3] - indices[2] + 1
            peak_conf[indices[0]:indices[1]+1, indices[2]:indices[3]+1] = np.zeros((search_width, search_height))
            peak_class[indices[0]:indices[1]+1, indices[2]:indices[3]+1] = - np.ones((search_width, search_height))
            # write the detected objects to matrix
            peak_conf[row_id, col_id] = conf_valu
            peak_class[row_id, col_id] = class_ids[o_class]

    return peak_conf, peak_class


def find_greatest_points(peak_conf, peak_class):

    detect_mat = - np.ones((rodnet_configs['max_dets'], 4))
    height = peak_conf.shape[0]
    width = peak_conf.shape[1]
    peak_flatten = peak_conf.flatten()
    indic = np.argsort(peak_flatten)
    ind_len = indic.shape[0]

    if ind_len >= rodnet_configs['max_dets']:
        choos_ind = np.flip(indic[-rodnet_configs['max_dets']:ind_len])
    else:
        choos_ind = np.flip(indic)

    for count, ele_ind in enumerate(choos_ind):
        row = ele_ind // width
        col = ele_ind % width
        if peak_conf[row, col] > 0:
            detect_mat[count, 0] = peak_class[row, col]
            detect_mat[count, 1] = row
            detect_mat[count, 2] = col
            detect_mat[count, 3] = peak_conf[row, col]

    return detect_mat


def post_processing(confmaps, peak_thres=0.1):
    """
    Post-processing for RODNet
    :param confmaps: predicted confidence map [B, n_class, win_size, ramap_r, ramap_a]
    :param search_size: search other detections within this window (resolution of our system)
    :param peak_thres: peak threshold
    :return: [B, win_size, max_dets, 4]
    """
    # print(confmaps.shape)
    batch_size = confmaps.shape[0]
    class_size = confmaps.shape[1]
    win_size = confmaps.shape[2]
    height = confmaps.shape[3]
    width = confmaps.shape[4]

    if class_size != n_class:
        raise TypeError("Wrong class number setting. ")

    max_dets = rodnet_configs['max_dets']

    rng_grid = confmap2ra(radar_configs, 'range')
    agl_grid = confmap2ra(radar_configs, 'angle')

    res_final = - np.ones((batch_size, win_size, max_dets, 4))

    for b in range(batch_size):
        for w in range(win_size):
            detect_mat = []
            for c in range(class_size):
                obj_dicts_in_class = []
                confmap = np.squeeze(confmaps[b, c, w, :, :])
                # detect peak
                rowids, colids = detect_peaks(confmap, threshold=peak_thres)

                for ridx, aidx in zip(rowids, colids):
                    rng = rng_grid[ridx]
                    agl = agl_grid[aidx]
                    conf = confmap[ridx, aidx]
                    obj_dict = {'frameid': None, 'range': rng, 'angle': agl, 'ridx': ridx, 'aidx': aidx,
                                'classid': c, 'score': conf}
                    obj_dicts_in_class.append(obj_dict)

                detect_mat_in_class = lnms(obj_dicts_in_class)
                detect_mat.append(detect_mat_in_class)

            detect_mat = np.array(detect_mat)
            detect_mat = np.reshape(detect_mat, (class_size*max_dets, 4))
            detect_mat = detect_mat[detect_mat[:, 3].argsort(kind='mergesort')[::-1]]
            res_final[b, w, :, :] = detect_mat[:max_dets]

    return res_final

def post_processing_single_timestamp(confmaps, peak_thres=0.1):
    """
    Post-processing for RODNet
    :param confmaps: predicted confidence map [B, n_class, win_size, ramap_r, ramap_a]
    :param search_size: search other detections within this window (resolution of our system)
    :param peak_thres: peak threshold
    :return: [B, win_size, max_dets, 4]
    """

    class_size = confmaps.shape[0]
    height = confmaps.shape[1]
    width = confmaps.shape[2]

    if class_size != n_class:
        raise TypeError("Wrong class number setting. ")

    max_dets = rodnet_configs['max_dets']

    rng_grid = confmap2ra(radar_configs, 'range')
    agl_grid = confmap2ra(radar_configs, 'angle')

    res_final = - np.ones((max_dets, 4))

    detect_mat = []
    for c in range(class_size):
        obj_dicts_in_class = []
        confmap = confmaps[c, :, :]
        # detect peak
        rowids, colids = detect_peaks(confmap, threshold=peak_thres)

        for ridx, aidx in zip(rowids, colids):
            rng = rng_grid[ridx]
            agl = agl_grid[aidx]
            conf = confmap[ridx, aidx]
            obj_dict = {'frameid': None, 'range': rng, 'angle': agl, 'ridx': ridx, 'aidx': aidx,
                        'classid': c, 'score': conf}
            obj_dicts_in_class.append(obj_dict)

        detect_mat_in_class = lnms(obj_dicts_in_class)
        detect_mat.append(detect_mat_in_class)

    detect_mat = np.array(detect_mat)
    detect_mat = np.reshape(detect_mat, (class_size*max_dets, 4))
    detect_mat = detect_mat[detect_mat[:, 3].argsort(kind='mergesort')[::-1]]
    res_final[:, :] = detect_mat[:max_dets]

    return res_final

def lnms(obj_dicts_in_class):

    detect_mat = - np.ones((rodnet_configs['max_dets'], 4))
    cur_det_id = 0
    # sort peaks by confidence score
    inds = np.argsort([-d['score'] for d in obj_dicts_in_class], kind='mergesort')
    dts = [obj_dicts_in_class[i] for i in inds]
    while len(dts) != 0:
        if cur_det_id >= rodnet_configs['max_dets']:
            break
        p_star = dts[0]
        detect_mat[cur_det_id, 0] = p_star['classid']
        detect_mat[cur_det_id, 1] = p_star['ridx']
        detect_mat[cur_det_id, 2] = p_star['aidx']
        detect_mat[cur_det_id, 3] = p_star['score']
        cur_det_id += 1
        del dts[0]
        for pid, pi in enumerate(dts):
            ols = get_ols_btw_objects(p_star, pi)
            if ols > rodnet_configs['ols_thres']:
                del dts[pid]

    return detect_mat


def write_dets_results(res, data_id, save_path):
    # batch_size = 1 when testing
    batch_size = 1
    with open(save_path, 'a+') as f:
        for b in range(batch_size):
            for w in range(rodnet_configs['win_size']):
                for d in range(rodnet_configs['max_dets']):
                    cla_id = int(res[b, w, d, 0])
                    if cla_id == -1:
                        continue
                    row_id = res[b, w, d, 1]
                    col_id = res[b, w, d, 2]
                    conf = res[b, w, d, 3]
                    f.write("%010d %s %d %d %s\n" % (data_id+w, class_table[cla_id], row_id, col_id, conf))


def write_dets_results_single_timestamp(res, data_id, save_path):
    with open(save_path, 'a+') as f:
        for d in range(rodnet_configs['max_dets']):
            cla_id = int(res[d, 0])
            if cla_id == -1:
                continue
            row_id = res[d, 1]
            col_id = res[d, 2]
            conf = res[d, 3]
            f.write("%010d %s %d %d %s\n" % (data_id, class_table[cla_id], row_id, col_id, conf))


if __name__ == "__main__":
    input_test = np.random.random_sample((1, 3, 16, 122, 91))
    res_final = post_processing(input_test)
    for b in range(1):
        for w in range(16):
            confmaps = np.squeeze(input_test[b, :, w, :, :])
            visualize_postprocessing(confmaps, res_final[b, w, :, :])
