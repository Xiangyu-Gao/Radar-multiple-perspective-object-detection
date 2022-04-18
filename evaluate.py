import os
import math
import numpy as np
import argparse

from utils import pol2cart
from utils.mappings import confmap2ra
from utils.read_annotations import read_ra_labels_csv, read_3d_labels_txt
from utils.ols import get_ols_btw_objects
from utils.visualization import visualize_ols_hist
from utils.dataset_tools import calculate_frame_offset

from config import n_class, class_table, class_ids, object_sizes
from config import radar_configs, rodnet_configs
from config import test_sets

rng_grid = confmap2ra(radar_configs, 'range')
agl_grid = confmap2ra(radar_configs, 'angle')

olsThrs = np.around(np.linspace(0.5, 0.9, int(np.round((0.9 - 0.5) / 0.05) + 1), endpoint=True), decimals=2)
recThrs = np.around(np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01) + 1), endpoint=True), decimals=2)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate RODNet.')
    parser.add_argument('-md', '--modeldir', type=str, dest='model_dir',
                        help='file name to load trained model')
    parser.add_argument('-rd', '--resdir', type=str, dest='res_dir', default='./results/',
                        help='directory to save testing results')
    args = parser.parse_args()
    return args


def load_rodnet_res(filename):
    with open(filename, 'r') as df:
        data = df.readlines()

    n_frame = int(float(data[-1].rstrip().split()[0])) + 1
    dts = {(i, j): [] for i in range(n_frame) for j in range(n_class)}

    for id, line in enumerate(data):
        if line is not None:
            line = line.rstrip().split()
            frameid, class_str, ridx, aidx, conf = line
            frameid = int(frameid)
            classid = class_ids[class_str]
            ridx = int(ridx)
            aidx = int(aidx)
            conf = float(conf)
            if conf > 1:
                conf = 1
            if conf < rodnet_configs['ols_thres']:
                continue
            rng = rng_grid[ridx]
            agl = agl_grid[aidx]
            if rng > rodnet_configs['rr_max_eval'] or rng < rodnet_configs['rr_min_eval'] or \
                    agl > rodnet_configs['ra_max_eval'] or agl < rodnet_configs['ra_min_eval']:
                continue
            obj_dict = {'id': id+1, 'frameid': frameid, 'range': rng, 'angle': agl, 'ridx': ridx, 'aidx': aidx,
                        'classid': classid, 'score': conf}
            dts[frameid, classid].append(obj_dict)

    return dts, n_frame


def load_vgg_res(filename):
    with open(filename, 'r') as df:
        data = df.readlines()

    n_frame = int(float(data[-1].rstrip().split()[0])) + 1
    dts = {(i, j): [] for i in range(n_frame) for j in range(n_class)}

    for id, line in enumerate(data):
        if line is not None:
            line = line.rstrip().split()
            frameid, ridx, aidx, classid, conf = line
            frameid = int(frameid)
            classid = int(classid)
            if 'onrd' in filename and classid == 1:
                rad = np.random.uniform(0, 1)
                if rad > 0.6:
                    classid = 2
            ridx = int(ridx)
            aidx = int(aidx)
            conf = float(conf)
            if conf > 1:
                conf = 1
            # if conf < 0.5:
            #     continue
            rng = rng_grid[ridx]
            agl = agl_grid[aidx]
            if rng > rodnet_configs['rr_max_eval'] or rng < rodnet_configs['rr_min_eval'] or \
                    agl > rodnet_configs['ra_max_eval'] or agl < rodnet_configs['ra_min_eval']:
                continue
            obj_dict = {'id': id+1, 'frameid': frameid, 'range': rng, 'angle': agl, 'ridx': ridx, 'aidx': aidx,
                        'classid': classid, 'score': conf}
            dts[frameid, classid].append(obj_dict)

    return dts, n_frame


def convert_gt_for_eval(obj_info_list):
    rng_grid = confmap2ra(radar_configs, 'range')
    agl_grid = confmap2ra(radar_configs, 'angle')

    n_frame = len(obj_info_list)
    gts = {(i, j): [] for i in range(n_frame) for j in range(n_class)}
    id = 1
    for frameid, obj_info in enumerate(obj_info_list):
        # for each frame
        for obj in obj_info:
            ridx, aidx, classid = obj
            try:
                class_str = class_table[classid]
            except:
                continue
            rng = rng_grid[ridx]
            agl = agl_grid[aidx]
            if rng > rodnet_configs['rr_max_eval'] or rng < rodnet_configs['rr_min_eval'] or \
                    agl > rodnet_configs['ra_max_eval'] or agl < rodnet_configs['ra_min_eval']:
                continue
            obj_dict = {'id': id, 'frameid': frameid, 'range': rng, 'angle': agl, 'ridx': ridx, 'aidx': aidx,
                        'classid': classid, 'score': 1.0}
            gts[frameid, classid].append(obj_dict)  # TODO: assume frameid start from 0
            id += 1
    return gts, n_frame


def compute_ols_dts_gts(gts_dict, dts_dict, imgId, catId):
    gts = gts_dict[imgId, catId]
    dts = dts_dict[imgId, catId]
    inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
    dts = [dts[i] for i in inds]
    # TODO: uncomment the lines below when max detections per image is needed
    # if len(dts) > maxDets[-1]:
    #     dts = dts[:maxDets[-1]]
    if len(gts) == 0 or len(dts) == 0:
        return []
    olss = np.zeros((len(dts), len(gts)))
    # compute oks between each detection and ground truth object
    for j, gt in enumerate(gts):
        for i, dt in enumerate(dts):
            olss[i, j] = get_ols_btw_objects(gt, dt)
    return olss


def evaluate_img(gts_dict, dts_dict, imgId, catId, olss_dict, log=False):
    gts = gts_dict[imgId, catId]
    dts = dts_dict[imgId, catId]
    if len(gts) == 0 and len(dts) == 0:
        return None

    if log:
        olss_flatten = np.ravel(olss_dict[imgId, catId])
        print("Frame %d: %10s %s" % (imgId, class_table[catId], list(olss_flatten)))

    # TODO: uncomment the lines below when score is available
    # sort dt highest score first, sort gt ignore last
    # gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
    # gt = [gt[i] for i in gtind]
    dtind = np.argsort([-d['score'] for d in dts], kind='mergesort')
    dts = [dts[i] for i in dtind]
    # iscrowd = [int(o['iscrowd']) for o in gt]
    # load computed olss
    # olss = olss[imgId, catId][:, gtind] if len(olss[imgId, catId]) > 0 else self.olss[imgId, catId]

    # for g in gts:
        # if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
        #     g['_ignore'] = 1
        # else:
        #     g['_ignore'] = 0

    olss = olss_dict[imgId, catId]

    T = len(olsThrs)
    G = len(gts)
    D = len(dts)
    gtm = np.zeros((T, G))
    dtm = np.zeros((T, D))
    # gtIg = np.array([g['_ignore'] for g in gts])
    # dtIg = np.zeros((T, D))

    if not len(olss) == 0:  # TODO: if len() work?
        for tind, t in enumerate(olsThrs):
            for dind, d in enumerate(dts):
                # information about best match so far (m=-1 -> unmatched)
                iou = min([t, 1-1e-10])
                m = -1
                for gind, g in enumerate(gts):
                    # if this gt already matched, and not a crowd, continue
                    if gtm[tind, gind] > 0:
                        continue
                    # # if dt matched to reg gt, and on ignore gt, stop
                    # if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                    #     break
                    # continue to next gt unless better match made
                    # print(olss[dind, gind])
                    if olss[dind, gind] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou = olss[dind, gind]
                    m = gind
                # if match made store id of match for both dt and gt
                if m == -1:
                    # no gt matched
                    continue
                # dtIg[tind, dind] = gtIg[m]
                dtm[tind, dind] = gts[m]['id']
                gtm[tind, m] = d['id']
    # set unmatched detections outside of area range to ignore
    # a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
    # dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))

    # store results for given image and category
    return {
            'image_id':     imgId,
            'category_id':  catId,
            'dtIds':        [d['id'] for d in dts],
            'gtIds':        [g['id'] for g in gts],
            'dtMatches':    dtm,
            'gtMatches':    gtm,
            'dtScores':     [d['score'] for d in dts],
            # 'gtIgnore':     gtIg,
            # 'dtIgnore':     dtIg,
        }


def accumulate(evalImgs, n_frame, log=True):
    T = len(olsThrs)
    R = len(recThrs)
    K = n_class
    precision = -np.ones((T, R, K))  # -1 for the precision of absent categories
    recall = -np.ones((T, K))
    scores = -np.ones((T, R, K))
    n_objects = np.zeros((K, ))

    for classid in range(n_class):
        E = [evalImgs[i * n_class + classid] for i in range(n_frame)]
        E = [e for e in E if not e is None]
        if len(E) == 0:
            continue

        dtScores = np.concatenate([e['dtScores'] for e in E])
        # different sorting method generates slightly different results.
        # mergesort is used to be consistent as Matlab implementation.
        inds = np.argsort(-dtScores, kind='mergesort')
        dtScoresSorted = dtScores[inds]

        dtm = np.concatenate([e['dtMatches'] for e in E], axis=1)[:, inds]
        gtm = np.concatenate([e['gtMatches'] for e in E], axis=1)
        nd = dtm.shape[1]  # number of detections
        ng = gtm.shape[1]  # number of ground truth
        n_objects[classid] = ng

        if log:
            print("%10s: %4d dets, %4d gts" % (class_table[classid], dtm.shape[1], gtm.shape[1]))

        tps = np.array(dtm, dtype=bool)
        fps = np.logical_not(dtm)
        # tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
        # fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )
        tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
        fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

        for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
            tp = np.array(tp)
            fp = np.array(fp)
            rc = tp / (ng+np.spacing(1))
            pr = tp / (fp+tp+np.spacing(1))
            q  = np.zeros((R,))
            ss = np.zeros((R,))

            if nd:
                recall[t, classid] = rc[-1]
            else:
                recall[t, classid] = 0

            # numpy is slow without cython optimization for accessing elements
            # use python array gets significant speed improvement
            pr = pr.tolist(); q = q.tolist()

            for i in range(nd-1, 0, -1):
                if pr[i] > pr[i-1]:
                    pr[i-1] = pr[i]

            inds = np.searchsorted(rc, recThrs, side='left')
            try:
                for ri, pi in enumerate(inds):
                    q[ri] = pr[pi]
                    ss[ri] = dtScoresSorted[pi]
            except:
                pass
            precision[t, :, classid] = np.array(q)
            scores[t, :, classid] = np.array(ss)

    eval = {
            'counts': [T, R, K],
            'object_counts': n_objects,
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }
    return eval


def summarize(eval, gl=True):
    def _summarize(eval=eval, ap=1, olsThr=None):

        iStr = ' {:<18} {} @[ OLS={:<9} ] = {:0.4f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap==1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(olsThrs[0], olsThrs[-1]) \
            if olsThr is None else '{:0.2f}'.format(olsThr)

        object_counts = eval['object_counts']
        n_objects = np.sum(object_counts)
        # print(object_counts, n_objects)

        # aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        # mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxK]
            s = eval['precision']
            # IoU
            if olsThr is not None:
                t = np.where(olsThr == olsThrs)[0]
                s = s[t]
            s = s[:,:,:]
        else:
            # dimension of recall: [TxK]
            s = eval['recall']
            if olsThr is not None:
                t = np.where(olsThr == olsThrs)[0]
                s = s[t]
            s = s[:,:]
        # mean_s = np.mean(s[s>-1])
        mean_s = 0
        for classid in range(n_class):
            if ap == 1:
                s_class = s[:, :, classid]
                if len(s_class[s_class>-1]) == 0:
                    pass
                else:
                    mean_s += object_counts[classid] / n_objects * np.mean(s_class[s_class>-1])
            else:
                s_class = s[:, classid]
                if len(s_class[s_class>-1]) == 0:
                    pass
                else:
                    mean_s += object_counts[classid] / n_objects * np.mean(s_class[s_class>-1])

        print(iStr.format(titleStr, typeStr, iouStr, mean_s))
        return mean_s

    def _summarizeKps():
        stats = np.zeros((9,))
        stats[0] = _summarize(ap=1)
        stats[1] = _summarize(ap=1, olsThr=.5)
        stats[2] = _summarize(ap=1, olsThr=.6)
        stats[3] = _summarize(ap=1, olsThr=.7)
        stats[4] = _summarize(ap=1, olsThr=.8)
        stats[5] = _summarize(ap=1, olsThr=.9)
        stats[6] = _summarize(ap=0)
        stats[7] = _summarize(ap=0, olsThr=.5)
        stats[8] = _summarize(ap=0, olsThr=.75)
        return stats

    def _summarizeKps_cur():
        stats = np.zeros((2,))
        stats[0] = _summarize(ap=1)
        stats[1] = _summarize(ap=0)
        return stats

    if gl:
        summarize = _summarizeKps
    else:
        summarize = _summarizeKps_cur

    stats = summarize()

    return stats


if __name__ == '__main__':
    """
    Example:
        python evaluate.py -md C3D-20200904-001923 -rd ./results/
    """
    args = parse_args()
    data_root = '/mnt/nas_crdataset'
    rodnet_res_folder = args.res_dir

    if args.model_dir is None:
        raise ValueError("model_dir is needed!")
    model_name = args.model_dir

    root_dir = test_sets['root_dir']
    dates = test_sets['dates']
    seqs = test_sets['seqs']

    easy_sets = ['2019_05_28_pm2s012']
    evalImgs_all = []
    evalImgs_easy = []
    evalImgs_mid = []
    evalImgs_hard = []
    n_frames_all = 0
    n_frames_easy = 0
    n_frames_mid = 0
    n_frames_hard = 0
    ols_list = []

    seq_names = sorted(os.listdir(os.path.join(rodnet_res_folder, model_name)))
    for seq_name in seq_names:
        if seq_name[0:2] != '20':
            continue
        print('true seq')
        date = seq_name[:10]
        seq_path = os.path.join(data_root, date, seq_name)
        rodnet_res_path = os.path.join(rodnet_res_folder, model_name, seq_name, 'rod_res.txt')

        try:
            print(rodnet_res_path)
            dts_dict, n_frame_dts = load_rodnet_res(rodnet_res_path)
        except:
            dts_dict, n_frame_dts = load_vgg_res(rodnet_res_path)

        # gts_dict, n_frame_gts = load_gt_labels(GT_LABEL_NAME)
        obj_info_list = read_ra_labels_csv(seq_path)

        gts_dict, n_frame_gts = convert_gt_for_eval(obj_info_list)
        n_frame = min(n_frame_dts, n_frame_gts)

        olss_all = {(imgId, catId): compute_ols_dts_gts(gts_dict, dts_dict, imgId, catId) \
                    for imgId in range(n_frame)
                    for catId in range(n_class)}

        for olss in list(olss_all.values()):
            if len(olss) > 0:
                olss_max_gt = np.amax(olss, axis=0)
                cur_olss = list(np.ravel(np.squeeze(olss_max_gt)))
                ols_list.extend(cur_olss)

        evalImgs = [evaluate_img(gts_dict, dts_dict, imgId, catId, olss_all)
                    for imgId in range(n_frame)
                    for catId in range(n_class)]

        eval = accumulate(evalImgs, n_frame, log=False)
        stats = summarize(eval, gl=False)

        if seq_name in easy_sets:
            n_frames_easy += n_frame
            evalImgs_easy.extend(evalImgs)
        n_frames_all += n_frame
        evalImgs_all.extend(evalImgs)

    eval = accumulate(evalImgs_all, n_frames_all)
    stats = summarize(eval)

    eval = accumulate(evalImgs_easy, n_frames_easy)
    stats = summarize(eval)
    # eval = accumulate(evalImgs_mid, n_frames_mid)
    # stats = summarize(eval)
    # eval = accumulate(evalImgs_hard, n_frames_hard)
    # stats = summarize(eval)

    # visualize_ols_hist(ols_list)
