# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.font_manager import FontProperties

from utils import chirp_amp
from config import class_table
from config import radar_configs, rodnet_configs

fig = plt.figure(figsize=(8, 8))

fp = FontProperties(fname=r"/home/yzwang/Documents/fontawesome-free-5.12.0-desktop/otfs/solid-900.otf")
symbols = {
    'pedestrian': "\uf554",
    'cyclist': "\uf84a",
    'car': "\uf1b9",
}


def visualize_confmap(confmap, pps=[]):
    if len(confmap.shape) == 2:
        plt.imshow(confmap, origin='lower', aspect='auto')
        for pp in pps:
            plt.scatter(pp[1], pp[0], s=5, c='white')
        plt.show()
        return
    else:
        n_channel, _, _ = confmap.shape
    if n_channel == 3:
        confmap_viz = np.transpose(confmap, (1, 2, 0))
    elif n_channel > 3:
        confmap_viz = np.transpose(confmap[:3, :, :], (1, 2, 0))
        if n_channel == 4:
            confmap_noise = confmap[3, :, :]
            plt.imshow(confmap_noise, origin='lower', aspect='auto')
            plt.show()
    else:
        print("Warning: wrong shape of confmap!")
        return
    plt.imshow(confmap_viz, origin='lower', aspect='auto')
    for pp in pps:
        plt.scatter(pp[1], pp[0], s=5, c='white')
    plt.show()


def visualize_confmaps_cr(confmapc, confmapr, confmapcr, ppsc=[], ppsr=[], ppres=[], figname=None):
    fig = plt.figure(figsize=(8, 8))
    n_channel, nr, na = confmapc.shape
    fig_id = 1
    for class_id in range(n_channel):
        fig.add_subplot(n_channel, 3, fig_id)
        fig_id += 1
        plt.imshow(confmapc[class_id], origin='lower', aspect='auto')
        for pp in ppsc[class_id]:
            plt.scatter(pp[1], pp[0], s=5, c='white')
        plt.xlim(0, na)
        plt.ylim(0, nr)

        fig.add_subplot(n_channel, 3, fig_id)
        fig_id += 1
        plt.imshow(confmapr, origin='lower', aspect='auto')
        for pp in ppsr:
            plt.scatter(pp[1], pp[0], s=5, c='white')
        plt.xlim(0, na)
        plt.ylim(0, nr)

        fig.add_subplot(n_channel, 3, fig_id)
        fig_id += 1
        plt.imshow(confmapcr[class_id], origin='lower', aspect='auto', vmin=0, vmax=1)
        for pp in ppres[class_id]:
            plt.scatter(pp[1], pp[0], s=5, c='white')
        plt.xlim(0, na)
        plt.ylim(0, nr)

    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)
        plt.close(fig)


def visualize_radar_chirp(chirp):
    if radar_configs['data_type'] == 'RI' or radar_configs['data_type'] == 'RISEP':
        chirp_abs = np.sqrt(chirp[:, :, 0] ** 2 + chirp[:, :, 1] ** 2)
    elif radar_configs['data_type'] == 'AP' or radar_configs['data_type'] == 'APSEP':
        chirp_abs = chirp[:, :, 0]
    else:
        raise ValueError
    plt.imshow(chirp_abs)
    plt.show()


def visualize_radar_chirps(chirp):
    if radar_configs['data_type'] == 'RI' or radar_configs['data_type'] == 'RISEP':
        chirp_abs = np.sqrt(chirp[:, :, :, 0] ** 2 + chirp[:, :, :, 1] ** 2)
    elif radar_configs['data_type'] == 'AP' or radar_configs['data_type'] == 'APSEP':
        chirp_abs = chirp[:, :, :, 0]
    else:
        raise ValueError
    chirp_abs_avg = np.mean(chirp_abs, axis=0)
    plt.imshow(chirp_abs_avg)
    plt.show()


def visualize_train_img_old(fig_name, input_radar, output_confmap, confmap_gt):
    fig = plt.figure(figsize=(8, 4))
    img = input_radar
    fig.add_subplot(1, 3, 1)
    plt.imshow(img, vmin=0, vmax=1, origin='lower', aspect='auto')
    img = output_confmap
    fig.add_subplot(1, 3, 2)
    plt.imshow(img, vmin=0, vmax=1, origin='lower', aspect='auto')
    img = confmap_gt
    fig.add_subplot(1, 3, 3)
    plt.imshow(img, vmin=0, vmax=1, origin='lower', aspect='auto')
    plt.savefig(fig_name)
    plt.close(fig)


def visualize_train_img(fig_name, img_path, input_radar, output_confmap, confmap_gt):

    fig = plt.figure(figsize=(8, 8))
    img_data = mpimg.imread(img_path)

    fig.add_subplot(2, 2, 1)
    plt.imshow(img_data.astype(np.uint8))

    fig.add_subplot(2, 2, 2)
    plt.imshow(input_radar, origin='lower', aspect='auto')

    fig.add_subplot(2, 2, 3)
    output_confmap = np.transpose(output_confmap, (1, 2, 0))
    output_confmap[output_confmap < 0] = 0
    plt.imshow(output_confmap, vmin=0, vmax=1, origin='lower', aspect='auto')

    fig.add_subplot(2, 2, 4)
    confmap_gt = np.transpose(confmap_gt, (1, 2, 0))
    plt.imshow(confmap_gt, vmin=0, vmax=1, origin='lower', aspect='auto')

    plt.savefig(fig_name)
    plt.close(fig)


def visualize_test_img(fig_name, img_path, input_radar, confmap_pred, confmap_gt, res_final, sybl=False):

    img_data = mpimg.imread(img_path)
    # radar_data = mpimg.imread(radar_path)
    fig.add_subplot(2, 2, 1)
    plt.imshow(img_data.astype(np.uint8))
    plt.axis('off')
    plt.title("Image")

    fig.add_subplot(2, 2, 2)
    # plt.imshow(radar_data.astype(np.float))
    plt.imshow(input_radar, origin='lower', aspect='auto')
    plt.axis('off')
    plt.title("RA Heatmap")

    fig.add_subplot(2, 2, 3)
    confmap_pred = np.transpose(confmap_pred, (1, 2, 0))
    confmap_pred[confmap_pred < 0] = 0
    confmap_pred[confmap_pred > 1] = 1
    plt.imshow(confmap_pred, vmin=0, vmax=1, origin='lower', aspect='auto')
    for d in range(rodnet_configs['max_dets']):
        cla_id = int(res_final[d, 0])
        if cla_id == -1:
            continue
        row_id = res_final[d, 1]
        col_id = res_final[d, 2]
        conf = res_final[d, 3]
        conf = 1.0 if conf > 1 else conf
        cla_str = class_table[cla_id]
        if sybl:
            text = symbols[cla_str]
            plt.text(col_id, row_id+3, text, fontproperties=fp, color='white', size=20, ha="center")
        else:
            plt.scatter(col_id, row_id, s=10, c='white')
            text = cla_str + '\n%.2f'%conf
            plt.text(col_id+5, row_id, text, color='white', fontsize=10)
    plt.axis('off')
    plt.title("RODNet Detection")

    fig.add_subplot(2, 2, 4)
    confmap_gt = np.transpose(confmap_gt, (1, 2, 0))
    plt.imshow(confmap_gt, vmin=0, vmax=1, origin='lower', aspect='auto')
    plt.axis('off')
    plt.title("Ground Truth")

    plt.savefig(fig_name)
    # plt.pause(0.1)
    plt.clf()



def visualize_test_img_wo_gt(fig_name, img_path, input_radar, confmap_pred, res_final, sybl=False):
    fig.set_size_inches(12, 4)

    img_data = mpimg.imread(img_path)
    # radar_data = mpimg.imread(radar_path)

    fig.add_subplot(1, 3, 1)
    # plt.imshow(img_data.astype(np.uint8))
    plt.imshow(img_data[:img_data.shape[0] // 5 * 4, :, :].astype(np.uint8))
    plt.axis('off')
    plt.title("Image")

    fig.add_subplot(1, 3, 2)
    # plt.imshow(radar_data.astype(np.float))
    input_radar[input_radar < 0] = 0
    input_radar[input_radar > 1] = 1
    plt.imshow(input_radar, vmin=0, vmax=1, origin='lower', aspect='auto')
    plt.axis('off')
    plt.title("RAMap")

    fig.add_subplot(1, 3, 3)
    confmap_pred = np.transpose(confmap_pred, (1, 2, 0))
    confmap_pred[confmap_pred < 0] = 0
    confmap_pred[confmap_pred > 1] = 1
    plt.imshow(confmap_pred, vmin=0, vmax=1, origin='lower', aspect='auto')
    for d in range(rodnet_configs['max_dets']):
        cla_id = int(res_final[d, 0])
        if cla_id == -1:
            continue
        row_id = res_final[d, 1]
        col_id = res_final[d, 2]
        conf = res_final[d, 3]
        conf = 1.0 if conf > 1 else conf
        cla_str = class_table[cla_id]
        if sybl:
            text = symbols[cla_str]
            plt.text(col_id+2, row_id+2, text, fontproperties=fp, color='white', size=20)
        else:
            plt.scatter(col_id, row_id, s=10, c='white')
            text = cla_str + '\n%.2f'%conf
            plt.text(col_id+5, row_id, text, color='white', fontsize=10)
    plt.axis('off')
    plt.title("RODNet Detections")

    plt.savefig(fig_name)
    plt.pause(0.1)
    plt.clf()


def heatmap2rgb(heatmap):
    cmap = plt.get_cmap('jet')
    rgba_img = cmap(heatmap)
    rgb_img = np.delete(rgba_img, 3, 2)
    rgb_img = np.transpose(rgb_img, (2, 0, 1))
    return rgb_img


def visualize_postprocessing(confmaps, det_results):
    confmap_pred = np.transpose(confmaps, (1, 2, 0))
    plt.imshow(confmap_pred, vmin=0, vmax=1, origin='lower', aspect='auto')
    for d in range(rodnet_configs['max_dets']):
        cla_id = int(det_results[d, 0])
        if cla_id == -1:
            continue
        row_id = det_results[d, 1]
        col_id = det_results[d, 2]
        conf = det_results[d, 3]
        cla_str = class_table[cla_id]
        plt.scatter(col_id, row_id, s=50, c='white')
        plt.text(col_id+5, row_id, cla_str + '\n%.2f'%conf, color='white', fontsize=10, fontweight='black')
    plt.axis('off')
    plt.title("RODNet Detection")
    plt.show()


def visualize_ols_hist(olss_flatten):
    _ = plt.hist(olss_flatten, bins='auto')  # arguments are passed to np.histogram
    plt.title("OLS Distribution")
    plt.show()


def visualize_anno_ramap(chirp, obj_info, figname, viz=False):

    chirp_abs = chirp_amp(chirp)
    plt.figure()
    plt.imshow(chirp_abs, vmin=0, vmax=1, origin='lower')

    for obj in obj_info:
        rng_idx, agl_idx, class_id = obj
        if class_id >= 0:
            try:
                cla_str = class_table[class_id]
            except:
                continue
        else:
            continue
        plt.scatter(agl_idx, rng_idx, s=10, c='white')
        plt.text(agl_idx+5, rng_idx, cla_str, color='white', fontsize=10)

    if viz:
        plt.show()
    else:
        plt.savefig(figname)
        plt.close()


def visualize_fuse_crdets(chirp, obj_dicts, figname=None, viz=False):

    chirp_abs = chirp_amp(chirp)
    chirp_shape = chirp_abs.shape
    plt.figure()
    plt.imshow(chirp_abs, vmin=0, vmax=1, origin='lower')

    for obj_id, obj_dict in enumerate(obj_dicts):
        plt.scatter(obj_dict['angle_id'], obj_dict['range_id'], s=10, c='white')
        try:
            text = str(obj_dict['object_id']) + ' ' + obj_dict['class']
        except:
            text = str(obj_dict['object_id'])
        plt.text(obj_dict['angle_id']+5, obj_dict['range_id'], text, color='white', fontsize=10)

    plt.xlim(0, chirp_shape[1])
    plt.ylim(0, chirp_shape[0])

    if viz:
        plt.show()
    else:
        plt.savefig(figname)
        plt.close()


def visualize_fuse_crdets_compare(img_path, chirp, c_dicts, r_dicts, cr_dicts, figname=None, viz=False):

    chirp_abs = chirp_amp(chirp)
    chirp_shape = chirp_abs.shape
    fig_local = plt.figure()
    fig_local.set_size_inches(16, 4)

    fig_local.add_subplot(1, 4, 1)
    im = plt.imread(img_path)
    plt.imshow(im)

    fig_local.add_subplot(1, 4, 2)
    plt.imshow(chirp_abs, vmin=0, vmax=1, origin='lower')
    for obj_id, obj_dict in enumerate(c_dicts):
        plt.scatter(obj_dict['angle_id'], obj_dict['range_id'], s=10, c='white')
        try:
            obj_dict['object_id']
        except:
            obj_dict['object_id'] = ''
        try:
            text = str(obj_dict['object_id']) + ' ' + obj_dict['class']
        except:
            text = str(obj_dict['object_id'])
        plt.text(obj_dict['angle_id']+5, obj_dict['range_id'], text, color='white', fontsize=10)
    plt.xlim(0, chirp_shape[1])
    plt.ylim(0, chirp_shape[0])

    fig_local.add_subplot(1, 4, 3)
    plt.imshow(chirp_abs, vmin=0, vmax=1, origin='lower')
    for obj_id, obj_dict in enumerate(r_dicts):
        plt.scatter(obj_dict['angle_id'], obj_dict['range_id'], s=10, c='white')
        try:
            obj_dict['object_id']
        except:
            obj_dict['object_id'] = ''
        try:
            text = str(obj_dict['object_id']) + ' ' + obj_dict['class']
        except:
            text = str(obj_dict['object_id'])
        plt.text(obj_dict['angle_id']+5, obj_dict['range_id'], text, color='white', fontsize=10)
    plt.xlim(0, chirp_shape[1])
    plt.ylim(0, chirp_shape[0])

    fig_local.add_subplot(1, 4, 4)
    plt.imshow(chirp_abs, vmin=0, vmax=1, origin='lower')
    for obj_id, obj_dict in enumerate(cr_dicts):
        plt.scatter(obj_dict['angle_id'], obj_dict['range_id'], s=10, c='white')
        try:
            obj_dict['object_id']
        except:
            obj_dict['object_id'] = '%.2f' % obj_dict['confidence']
        try:
            text = str(obj_dict['object_id']) + ' ' + obj_dict['class']
        except:
            text = str(obj_dict['object_id'])
        plt.text(obj_dict['angle_id']+5, obj_dict['range_id'], text, color='white', fontsize=10)
    plt.xlim(0, chirp_shape[1])
    plt.ylim(0, chirp_shape[0])

    if viz:
        plt.show()
    else:
        plt.savefig(figname)
        plt.close()
