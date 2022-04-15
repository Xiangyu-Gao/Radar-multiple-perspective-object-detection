# directory settings
data_sets = {
    'root_dir': "/mnt/nas_crdataset2",
    # 'dates': [
    #     '2019_04_09', '2019_04_30', '2019_05_09', '2019_05_23',
    #     '2019_05_28', '2019_05_29',
    #     # '2019_07_18', '2019_07_25', '2019_08_01', '2019_08_14', '2019_08_30',
    #     '2019_09_08', '2019_09_18',
    #     '2019_09_29',
    # ],
    'dates': ['2019_09_29'],
    'cam_anno': [
        False, False, False, False,
        False, False,
        # True, True, True, True, True,
        True, True,
        True,
    ],
}
# train_sets = {
#     'root_dir': "/mnt/nas_crdataset",
#     'dates': ['2019_04_09', '2019_04_30', '2019_05_09', '2019_05_29', '2019_09_08', '2019_09_29'],
#     # 'dates': ['2019_09_29'],
#     'seqs': [
#         None,
#         None,
#         ['2019_05_09_bs1s004', '2019_05_09_bm1s007', '2019_05_09_cm1s003', '2019_05_09_cm1s004',
#          '2019_05_09_cs1s001', '2019_05_09_cs1m000', '2019_05_09_pbms004'],
#         ['2019_05_29_pm1s016', '2019_05_29_pm1s017', '2019_05_29_pm2s015', '2019_05_29_pm3s000',
#          '2019_05_29_ps1s006', '2019_05_29_bm1s016', '2019_05_29_bm1s017',
#          '2019_05_29_bm1s018', '2019_05_29_bs1s007', '2019_05_29_cm1s014', '2019_05_29_cm1s015',
#          '2019_05_29_cm1s016', '2019_05_29_cm1s017', '2019_05_29_bcms000', '2019_05_29_pbms007',
#          '2019_05_29_pcms005', '2019_05_29_mlms006'], #  load error
#         None,
#         ['2019_09_29_onrd000', '2019_09_29_onrd010', '2019_09_29_onrd020', '2019_09_29_onrd031',
#          '2019_09_29_onrd001', '2019_09_29_onrd011', '2019_09_29_onrd021', '2019_09_29_onrd032',
#          '2019_09_29_onrd002', '2019_09_29_onrd012', '2019_09_29_onrd022', '2019_09_29_onrd033',
#          '2019_09_29_onrd003', '2019_09_29_onrd013', '2019_09_29_onrd023', '2019_09_29_onrd034',
#          '2019_09_29_onrd004', '2019_09_29_onrd014', '2019_09_29_onrd024', '2019_09_29_onrd035',
#          '2019_09_29_onrd005', '2019_09_29_onrd015', '2019_09_29_onrd025',
#          '2019_09_29_onrd006', '2019_09_29_onrd016', '2019_09_29_onrd026', '2019_09_29_onrd039',
#          '2019_09_29_onrd007', '2019_09_29_onrd017', '2019_09_29_onrd027', '2019_09_29_onrd040',
#          '2019_09_29_onrd008', '2019_09_29_onrd018', '2019_09_29_onrd028', '2019_09_29_onrd041',
#          '2019_09_29_onrd009', '2019_09_29_onrd030', '2019_09_29_onrd042' ],  # TODO: add sequence names
#         # ['2019_09_29_onrd014']
#     ],  # 'seqs' is two list corresponding to 'dates', include all seqs if None
#     'cam_anno': [False, False, False, False, True, True],
#     # 'cam_anno': [True],
# }  # training files

# train_sets = {
#     'root_dir': "/mnt/nas_crdataset",
#     'dates': ['2019_09_08', '2019_09_29'],
#     # 'dates': ['2019_09_29'],
#     'seqs': [
#         None,
#         ['2019_09_29_onrd000', '2019_09_29_onrd010', '2019_09_29_onrd020', '2019_09_29_onrd031',
#          '2019_09_29_onrd001', '2019_09_29_onrd011', '2019_09_29_onrd021', '2019_09_29_onrd032',
#          '2019_09_29_onrd002', '2019_09_29_onrd012', '2019_09_29_onrd022', '2019_09_29_onrd033',
#          '2019_09_29_onrd003', '2019_09_29_onrd013', '2019_09_29_onrd023', '2019_09_29_onrd034',
#          '2019_09_29_onrd004', '2019_09_29_onrd014', '2019_09_29_onrd024', '2019_09_29_onrd035',
#          '2019_09_29_onrd005', '2019_09_29_onrd015', '2019_09_29_onrd025',
#          '2019_09_29_onrd006', '2019_09_29_onrd016', '2019_09_29_onrd026', '2019_09_29_onrd039',
#          '2019_09_29_onrd007', '2019_09_29_onrd017', '2019_09_29_onrd027', '2019_09_29_onrd040',
#          '2019_09_29_onrd008', '2019_09_29_onrd018', '2019_09_29_onrd028', '2019_09_29_onrd041',
#          '2019_09_29_onrd009', '2019_09_29_onrd030', '2019_09_29_onrd042' ],  # TODO: add sequence names
#         # ['2019_09_29_onrd014']
#     ],  # 'seqs' is two list corresponding to 'dates', include all seqs if None
#     'cam_anno': [True, True],
#     # 'cam_anno': [True],
# }  # training files
#
# train_sets = {
#     'root_dir': "/mnt/nas_crdataset",
#     'dates': ['2019_04_09', '2019_04_30', '2019_05_09', '2019_05_29'],
#     # 'dates': ['2019_05_29'],
#     'seqs': [
#         None,
#         None,
#         ['2019_05_09_bs1s004', '2019_05_09_bm1s007', '2019_05_09_cm1s003', '2019_05_09_cm1s004',
#          '2019_05_09_cs1s001', '2019_05_09_cs1m000', '2019_05_09_pbms004'],
#         ['2019_05_29_pm1s016', '2019_05_29_pm1s017', '2019_05_29_pm2s015', '2019_05_29_pm3s000',
#          '2019_05_29_ps1s006', '2019_05_29_bm1s016', '2019_05_29_bm1s017',
#          '2019_05_29_bm1s018', '2019_05_29_bs1s007', '2019_05_29_cm1s014', '2019_05_29_cm1s015',
#          '2019_05_29_cm1s016', '2019_05_29_cm1s017', '2019_05_29_bcms000', '2019_05_29_pbms007',
#          '2019_05_29_pcms005', '2019_05_29_mlms006'], #  load error
#     ],  # 'seqs' is two list corresponding to 'dates', include all seqs if None
#     'cam_anno': [False, False, False, False],
#     # 'cam_anno': [True],
# }  # training files

train_sets = {
    'root_dir': "/mnt/nas_crdataset",
    'dates': ['2019_09_29'],
    # 'dates': ['2019_05_29'],
    'seqs': [
        ['2019_09_29_onrd000', '2019_09_29_onrd001', '2019_09_29_onrd003',
         '2019_09_29_onrd004', '2019_09_29_onrd017', '2019_09_29_onrd018'],
        # ['2019_09_29_onrd004'],
    ],  # 'seqs' is two list corresponding to 'dates', include all seqs if None
    'cam_anno': [True],
}  # training files


valid_sets = {
    'root_dir': "/mnt/nas_crdataset",
    'dates': ['2019_05_28'],
    'seqs': [
        ['2019_05_28_mlms005']
    ],  # 'seqs' is two list corresponding to 'dates', include all seqs if None
}  # validation files
# test_sets = {
#     'root_dir': "/mnt/nas_crdataset",
#     'dates': ['2019_05_28', '2019_09_18'],
#     # 'dates': ['2019_05_28'],
#     # 'dates': ['2019_09_18'],
#     # 'dates': ['2019_10_13'],
#     'seqs': [
#         ['2019_05_28_bm1s011', '2019_05_28_bm1s012', '2019_05_28_bm1s013', '2019_05_28_bm1s014', '2019_05_28_bs1s006',
#          '2019_05_28_cm1s009', '2019_05_28_cm1s010', '2019_05_28_cm1s011', '2019_05_28_cm1s012', '2019_05_28_cm1s013',
#          '2019_05_28_cs1s004', '2019_05_28_cs1s005', '2019_05_28_cs1s006', '2019_05_28_mlms005', '2019_05_28_pbms006',
#          '2019_05_28_pcms004', '2019_05_28_pm2s012', '2019_05_28_pm2s013', '2019_05_28_pm2s014', ],
#         # [
#         #     '2019_05_28_cm1s009', '2019_05_28_cm1s012', '2019_05_28_cm1s013',
#         #     '2019_05_28_cs1s004',
#         #     '2019_05_28_pm2s012',
#         #     '2019_05_28_pm2s013', '2019_05_28_pm2s014',
#         # ],
#         ['2019_09_18_onrd004', '2019_09_18_onrd009', ],
#         # ['2019_09_18_onrd004', ],
#         # ['2019_10_13_onrd048'],
#     ],  # 'seqs' is two list corresponding to 'dates', include all seqs if None
#     'cam_anno': [False, True],
#     # 'cam_anno': [False],
#     # 'cam_anno': [True],
#     # 'cam_anno': [False],
# }  # test files

test_sets = {
    'root_dir': "/mnt/nas_crdataset",
    'dates': ['2019_05_28', '2019_09_18'],
    # 'dates': ['2019_05_28'],
    # 'dates': ['2019_09_18'],
    # 'dates': ['2019_10_13'],
    'seqs': [
        # ['2019_05_28_cm1s009', '2019_05_28_cm1s011', '2019_05_28_cm1s012',
        #  '2019_05_28_cm1s013', '2019_05_28_pbms006', '2019_05_28_cm1s010',
        #  '2019_05_28_mlms005', '2019_05_28_pcms004'],
        # [ '2019_05_28_cm1s010', '2019_05_28_mlms005', '2019_05_28_pcms004',
        #   '2019_05_28_cm1s009', '2019_05_28_cm1s011', '2019_05_28_cm1s012',
        #   '2019_05_28_cm1s013', '2019_05_28_pbms006', '2019_05_28_cs1s004',
        #   '2019_05_28_bs1s006'],
        # [ '2019_05_28_cm1s012', '2019_05_28_mlms005', '2019_05_28_pcms004',
        #   '2019_05_28_cm1s009', '2019_05_28_cm1s011', '2019_05_28_cm1s010'],
        # ['2019_05_28_cm1s013', '2019_05_28_pbms006', '2019_05_28_cs1s004',
        #  '2019_05_28_bs1s006'],
        # ['2019_09_18_onrd004', '2019_09_18_onrd009'],

        # ['2019_05_28_cm1s012'],
        # ['2019_05_28_cm1s012', '2019_05_28_mlms005', '2019_05_28_cm1s013'],
        # ['2019_05_28_cm1s009', '2019_05_28_cm1s011', '2019_05_28_cm1s010'],
        # ['2019_05_28_cs1s004', '2019_05_28_pcms004', '2019_05_28_pbms006'],
        ['2019_05_28_bs1s006'],
        ['2019_09_18_onrd004', '2019_09_18_onrd009'],

        # easy scenrio
        # ['2019_05_28_bm1s011', '2019_05_28_bm1s012', '2019_05_28_bm1s013', '2019_05_28_bm1s014'],
        # ['2019_05_28_cs1s005', '2019_05_28_cs1s006', '2019_05_28_pm2s012', '2019_05_28_pm2s013', '2019_05_28_pm2s014'],


    ],  # 'seqs' is two list corresponding to 'dates', include all seqs if None
    # 'cam_anno': [False, True],
    # 'cam_anno': [False],
    'cam_anno': [False, False],
    # 'cam_anno': [True],
}  # test files

supertest_sets = {
    'root_dir': "/mnt/nas_crdataset",
    'dates': ['2019_09_18', '2019_09_29'],
    'seqs': [
        ['2019_09_18_onrd000', '2019_09_18_onrd001', '2019_09_18_onrd004', '2019_09_18_onrd005', '2019_09_18_onrd006',
         '2019_09_18_onrd010', '2019_09_18_onrd012', '2019_09_18_onrd013'],
        ['2019_09_29_onrd004', '2019_09_29_onrd005', '2019_09_29_onrd006'],
    ],  # 'seqs' is two list corresponding to 'dates', include all seqs if None
}  # supertest files

# class settings
n_class = 3
class_table = {
    0: 'pedestrian',
    1: 'cyclist',
    2: 'car',
    # 3: 'van',
    # 4: 'truck',
}

class_ids = {
    'pedestrian': 0,
    'cyclist': 1,
    'car': 2,
    'truck': 2,  # TODO: due to detection model bug
    'train': 2,
    'noise': -1000,
}

confmap_sigmas = {
    'pedestrian': 15,
    'cyclist': 20,
    'car': 30,
    # 'van': 12,
    # 'truck': 20,
}

confmap_sigmas_interval = {
    'pedestrian': [5, 15],
    'cyclist': [8, 20],
    'car': [10, 30],
    # 'van': 12,
    # 'truck': 20,
}

confmap_length = {
    'pedestrian': 1,
    'cyclist': 2,
    'car': 3,
    # 'van': 12,
    # 'truck': 20,
}

object_sizes = {
    'pedestrian': 0.5,
    'cyclist': 1.0,
    'car': 3.0,
}

# calibration
t_cl2cr = [0.35, 0, 0]
t_cl2rh = [0.11, -0.05, 0.06]
t_cl2rv = [0.21, -0.05, 0.06]

# parameter settings
camera_configs = {
    'image_width': 1440,
    'image_height': 1080,
    'frame_rate': 30,
    # 'image_folder': 'images_0',
    # 'image_folder': 'images_hist_0',
    'image_folder': 'images',
    'time_stamp_name': 'timestamps.txt',
    # 'time_stamp_name': 'timestamps_0.txt',
    'frame_expo': 0,
    # 'frame_expo': 40,
    'start_time_name': 'start_time.txt',
}
radar_configs = {
    'ramap_rsize': 128,             # RAMap range size
    'ramap_asize': 128,             # RAMap angle size
    'ramap_vsize': 128,             # RAMap angle size
    'frame_rate': 30,
    'crop_num': 3,                  # crop some indices in range domain
    'n_chirps': 255,                # number of chirps in one frame
    'sample_freq': 4e6,
    'sweep_slope': 21.0017e12,
    'data_type': 'RISEP',           # 'RI': real + imaginary, 'AP': amplitude + phase
    'ramap_rsize_label': 122,       # TODO: to be updated
    'ramap_asize_label': 121,       # TODO: to be updated
    'ra_min_label': -60,            # min radar angle
    'ra_max_label': 60,             # max radar angle
    'rr_min': 1.0,                  # min radar range (fixed)
    'rr_max': 25.0,                 # max radar range (fixed)
    'ra_min': -90,                  # min radar angle (fixed)
    'ra_max': 90,                   # max radar angle (fixed)
    'ramap_folder': 'WIN_HEATMAP',
}

# network settings
rodnet_configs = {
    'data_folder': 'WIN_PROC_MAT_DATA',
    # 'label_folder': 'dets_3d',
    'label_folder': 'dets_refine',
    'n_epoch': 100,
    'batch_size': 3,
    'learning_rate': 1e-5,
    'lr_step': 5,       # lr will decrease 10 times after lr_step epoches
    'win_size': 16,
    'input_rsize': 128,
    'input_asize': 128,
    'rr_min': 1.0,                  # min radar range
    'rr_max': 24.0,                 # max radar range
    'ra_min': -90.0,                  # min radar angle
    'ra_max': 90.0,                   # max radar angle
    'rr_min_eval': 1.0,                  # min radar range
    'rr_max_eval': 20.0,                 # max radar range
    'ra_min_eval': -60.0,                  # min radar angle
    'ra_max_eval': 60.0,                   # max radar angle
    'max_dets': 20,
    'peak_thres': 0.2,
    'ols_thres': 0.2,
    'stacked_num': 2,
    'test_stride': 2,
}

semi_loss_err_reg = {
    # index unit
    'level1': 30,
    'level2': 60,
    'level3': 80,
}
# correct error region for level 1
err_cor_reg_l1 = {
    'top': 3,
    'bot': 3,
}
# correct error region for level 2
err_cor_reg_l2 = {
    'top': 3,
    'bot': 25,
}
# correct error region for level 3
err_cor_reg_l3 = {
    'top': 3,
    'bot': 35,
}