import os
import time
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from postprocess import post_processing, post_processing_single_timestamp, write_dets_results_single_timestamp
from utils import str2bool
from utils import chirp_amp
from utils.dataset_tools import fix_cam_drop_frames
from utils.visualization import visualize_test_img, visualize_test_img_wo_gt
from config import n_class, test_sets, camera_configs, radar_configs, rodnet_configs
from config import mean1, std1, mean2, std2, mean1_rv, std1_rv, mean2_rv, std2_rv, mean1_va, std1_va, mean2_va, std2_va

CAM_DROP_FRAME = True
CAM_DELAY = False
Norm = True


def parse_args():
    parser = argparse.ArgumentParser(description='Test RODNet.')
    parser.add_argument('-m', '--model', type=str, dest='model',
                        help='choose rodnet model')
    parser.add_argument('-md', '--modeldir', type=str, dest='model_dir',
                        help='file name to load trained model')
    parser.add_argument('-dd', '--datadir', type=str, dest='data_dir', default='./data/',
                        help='directory to load data')
    parser.add_argument('-ld', '--logdir', type=str, dest='log_dir', default='./results/',
                        help='directory to save trained model')
    parser.add_argument('-rd', '--resdir', type=str, dest='res_dir', default='./results/',
                        help='directory to save testing results')
    parser.add_argument('-d', '--demo', type=str2bool, dest='demo', default=False,
                        help='False: test with GT, True: demo without GT')
    parser.add_argument('-s', '--symbol', type=str2bool, dest='sybl', default=False,
                        help='use symbol or text+score')
    args = parser.parse_args()
    return args


class GenConfmap():
    def __init__(self):
        self.confmap = np.zeros((n_class, radar_configs['ramap_rsize'], radar_configs['ramap_asize']))
        self.count = 0
        self.next = None
        self.ready = False

    def append(self, confmap):
        self.confmap = (self.confmap * self.count + confmap) / (self.count + 1)
        self.count += 1

    def setNext(self, _genconfmap):
        self.next = _genconfmap


if __name__ == "__main__":

    """
    Example:
        python test.py -m C3D -md C3D-20200904-001923
    """
    args = parse_args()
    sybl = args.sybl

    if args.model == 'CDC':
        from model.RODNet_CDC import RODNet
        from dataLoader.CRDatasets_ra import CRDataset
    elif args.model == 'HG':
        from model.RODNet_HG import RODNet
        from dataLoader.CRDatasets_ra import CRDataset
    elif args.model == 'C3D':
        from model.RODNet_3D import RODNet
        from dataLoader.CRDatasets import CRDataset
    else:
        raise TypeError

    if args.model_dir is not None and os.path.exists(os.path.join(args.log_dir, args.model_dir)):
        model_dir = args.model_dir
        models_loaded = sorted(os.listdir(os.path.join(args.log_dir, model_dir)))
        for fid, file in enumerate(models_loaded):
            if not file.endswith('.pkl'):
                del models_loaded[fid]
        if len(models_loaded) == 0:
            raise ValueError("No trained model found.")
        else:
            model_name = models_loaded[-1]
            print(model_name)
    else:
        raise ValueError("No trained model found.")

    win_size = rodnet_configs['win_size']
    if args.model == 'HG':
        rodnet = RODNet(n_class=n_class, win_size=win_size, stacked_num=rodnet_configs['stacked_num']).cuda()
        stacked_num = 2
    else:
        rodnet = RODNet(n_class=n_class, win_size=win_size).cuda()
        stacked_num = 1
    checkpoint = torch.load(os.path.join(os.path.join(args.log_dir, model_dir, model_name)))
    if 'optimizer_state_dict' in checkpoint:
        rodnet.load_state_dict(checkpoint['model_state_dict'])
    else:
        rodnet.load_state_dict(checkpoint)
    rodnet.eval()

    test_res_dir = os.path.join(os.path.join(args.res_dir, model_dir))
    if not os.path.exists(test_res_dir):
        os.makedirs(test_res_dir)
    
    total_time = 0
    total_count = 0

    root_dir = test_sets['root_dir']
    dates = test_sets['dates']
    seqs = test_sets['seqs']
    seq_names = []
    for dateid, date in enumerate(dates):
        seqs_cur = test_sets['seqs'][dateid]
        if seqs_cur is None:
            seqs_cur = sorted(os.listdir(os.path.join(root_dir, date)))
        seq_names.extend(seqs_cur)
    print(seq_names)

    for subset in seq_names:
        print(subset)
        if not args.demo:
            crdata_test = CRDataset(os.path.join(args.data_dir, 'data_details'),
                                    os.path.join(args.data_dir, 'confmaps_gt'),
                                    win_size=win_size, set_type='test',
                                    stride=rodnet_configs['test_stride'], subset=subset)
        else:
            crdata_test = CRDataset(os.path.join(args.data_dir, 'data_details'),
                                    None, win_size=win_size, set_type='supertest',
                                    stride=rodnet_configs['test_stride'], subset=subset)
        print("Length of testing data: %d" % len(crdata_test))

        dataloader = DataLoader(crdata_test, batch_size=1, shuffle=False, num_workers=0)
        seq_names = crdata_test.seq_names
        index_mapping = crdata_test.index_mapping

        for seq_name in seq_names:
            seq_res_dir = os.path.join(test_res_dir, seq_name)
            if not os.path.exists(seq_res_dir):
                os.makedirs(seq_res_dir)
            seq_res_viz_dir = os.path.join(seq_res_dir, 'rod_viz')
            if not os.path.exists(seq_res_viz_dir):
                os.makedirs(seq_res_viz_dir)
            f = open(os.path.join(seq_res_dir, 'rod_res.txt'), 'w')
            f.close()

        init_genConfmap = GenConfmap()
        iter_ = init_genConfmap

        for i in range(rodnet_configs['win_size'] - 1):
            while iter_.next != None:
                iter_ = iter_.next
            iter_.next = GenConfmap()

        load_tic = time.time()

        for iter, loaded_data in enumerate(dataloader):
            print("loading time: %.2f" % (time.time() - load_tic))
            if not args.demo:
                if args.model == 'C3D':
                    data, data_rv, data_va, confmap_gt, obj_info, dataloader_id = loaded_data
                else:
                   data, confmap_gt, obj_info, dataloader_id = loaded_data
            else:
                data, data_rv, data_va, data_rcs, dataloader_id = loaded_data
                confmap_gt = None
                obj_info = None

            seq_id, data_id, ra_frame_offset = index_mapping[dataloader_id]
            seq_name = seq_names[seq_id]
            seq_path = os.path.join(test_sets['root_dir'], seq_name[:10], seq_name)
            if int(seq_name[5:7]) < 9 and int(seq_name[0:4]) == 2019:
                image_folder = 'images'
                if Norm:
                    data = (data - mean1) / std1
                    print('finished ra normalization')
                    if args.model == 'C3D':
                        print('finished v normalization')
                        data_rv = (data_rv - mean1_rv) / std1_rv
                        data_va = (data_va - mean1_va) / std1_va
            elif int(seq_name[5:7]) >= 9 and int(seq_name[0:4]) == 2019:
                image_folder = 'images_0'
                if Norm:
                    data = (data - mean2) / std2
                    if args.model == 'C3D':
                        data_rv = (data_rv - mean2_rv) / std2_rv
                        data_va = (data_va - mean2_va) / std2_va

            cam_delay = ra_frame_offset
            try:
                img_paths = sorted(os.listdir(os.path.join(seq_path, image_folder)))
            except:
                img_paths = sorted(os.listdir(os.path.join(seq_path, 'images')))


            if CAM_DROP_FRAME:
                img_paths_fixed = fix_cam_drop_frames(seq_path, img_paths)
            else:
                img_paths_fixed = img_paths

            save_path = os.path.join(test_res_dir, seq_name, 'rod_res.txt')
            print("Testing %s/%06d-%06d... (%d)" % (seq_name, ra_frame_offset + data_id, ra_frame_offset + data_id+win_size, cam_delay))

            tic = time.time()

            if args.model == 'C3D':
                confmap_pred, _ = rodnet(data.float().cuda(), data_rv.float().cuda(), data_va.float().cuda())
            else:
                confmap_pred = rodnet(data.float().cuda())
            if stacked_num > 1:
                confmap_pred = confmap_pred[stacked_num - 1]

            confmap_pred = confmap_pred[-1].cpu().detach().numpy()
            confmap_pred = np.expand_dims(confmap_pred, axis=0)
            total_time += time.time() - tic
            print("%s/%010d.jpg inference finished in %.4f seconds." % (seq_names[seq_id], ra_frame_offset +data_id, time.time() - tic))

            res_final_once = post_processing(confmap_pred, rodnet_configs['peak_thres'])

            iter_ = init_genConfmap
            for i in range(confmap_pred.shape[2]):
                if iter_.next is None and i != confmap_pred.shape[2] - 1:
                    iter_.next = GenConfmap()
                iter_.append(confmap_pred[0, :, i, :, :])
                iter_ = iter_.next

            process_tic = time.time()
            radar_input_win = data.numpy()

            for i in range(rodnet_configs['test_stride']):
                total_count += 1
                imgid = data_id + i + camera_configs['frame_expo']
                radid = ra_frame_offset + data_id + i
                res_final = post_processing_single_timestamp(init_genConfmap.confmap, rodnet_configs['peak_thres'])
                write_dets_results_single_timestamp(res_final, radid, save_path)
                confmap_pred_0 = init_genConfmap.confmap
                res_final_0 = res_final
                img_path = os.path.join(seq_path, image_folder, img_paths_fixed[imgid])

                radar_input = chirp_amp(radar_input_win[0, :, i, :, :])
                fig_name = os.path.join(test_res_dir, seq_name, 'rod_viz', '%010d.jpg' % (imgid))

                if confmap_gt is not None:
                    confmap_gt_0 = confmap_gt[0, :, i, :, :]
                    visualize_test_img(fig_name, img_path, radar_input, confmap_pred_0, confmap_gt_0, res_final_0, sybl=sybl)
                else:
                    visualize_test_img_wo_gt(fig_name, img_path, radar_input, confmap_pred_0, res_final_0, sybl=sybl)
                init_genConfmap = init_genConfmap.next

            if iter == len(dataloader) - 1:
                offset = rodnet_configs['test_stride']
                imgid = data_id + offset + camera_configs['frame_expo']
                radid = ra_frame_offset + data_id + offset

                while init_genConfmap is not None:
                    total_count += 1
                    res_final = post_processing_single_timestamp(init_genConfmap.confmap, rodnet_configs['peak_thres'])
                    write_dets_results_single_timestamp(res_final, radid, save_path)
                    confmap_pred_0 = init_genConfmap.confmap
                    res_final_0 = res_final
                    img_path = os.path.join(seq_path, image_folder, img_paths_fixed[imgid])
                    radar_input = chirp_amp(radar_input_win[0, :, offset, :, :])
                    fig_name = os.path.join(test_res_dir, seq_name, 'rod_viz', '%010d.jpg' % (imgid))

                    if confmap_gt is not None:
                        confmap_gt_0 = confmap_gt[0, :, offset, :, :]
                        visualize_test_img(fig_name, img_path, radar_input, confmap_pred_0, confmap_gt_0, res_final_0, sybl=sybl)
                    else:
                        visualize_test_img_wo_gt(fig_name, img_path, radar_input, confmap_pred_0, res_final_0, sybl=sybl)

                    init_genConfmap = init_genConfmap.next
                    offset += 1
                    imgid += 1
                    radid += 1

            if init_genConfmap is None:
                init_genConfmap = GenConfmap()

            print("processing time: %.2f" % (time.time() - process_tic))
            load_tic = time.time()

    print("ave time: %f" % (total_time / total_count))
