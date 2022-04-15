import numpy as np
import os
import scipy.io as spio
import shutil
import pickle
import torch

from random import randint, random
from utils.confidence_map import generate_confmap, normalize_confmap, add_noise_channel
from relocate_dataset import produce_RV_slice, produce_RA_slice, produce_VA_slice
from utils.mappings import confmap2ra
from config import radar_configs
from utils.read_annotations import read_ra_labels_csv
from data_aug import transition_angle, transition_range
from cal_vel import velocity_fft
n_class = 3
viz = False


def find_obj_info(obj_class, obj_info):
    if obj_class == 'pedestrian':
        class_id = 0
    elif obj_class == 'cyclist':
        class_id = 1
    elif obj_class == 'car':
        class_id = 2
    else:
        pass
    # find the first occurence of class id
    need_info = None
    for info in obj_info:
        if info[2] == class_id:
            need_info = info
            break
    return need_info


def translation(RA_slice, RV_slice, VA_slice, confmap_gt, shift_range, shift_angle):
    data_ra = np.transpose(RA_slice, (3, 2, 0, 1))  # new dimesinon [C, chirps, range, angle]
    datara_shape = data_ra.shape
    data_ra = np.reshape(data_ra, (1, datara_shape[0], datara_shape[1], datara_shape[2], datara_shape[3]))
    data_ra = torch.from_numpy(data_ra)
    # new dimesinon [1, C, chirps, range, angle]

    data_rv = np.transpose(RV_slice, (2, 0, 1))  # new dimesinon [2chirps, range, velocity]
    datarv_shape = data_rv.shape
    data_rv = np.reshape(data_rv, (1, 1, datarv_shape[0], datarv_shape[1], datarv_shape[2]))
    data_rv = torch.from_numpy(data_rv)
    # new dimesinon [1, 1, 2chirps, range, velocity]

    data_va = np.transpose(VA_slice, (2, 0, 1))  # new dimesinon [2chirps, angle, velocity]
    datava_shape = data_va.shape
    data_va = np.reshape(data_va, (1, 1, datava_shape[0], datava_shape[1], datava_shape[2]))
    data_va = torch.from_numpy(data_va)
    # new dimesinon [1, 1, 2chirps, angle, velocity]

    confmap_shape = confmap_gt.shape  # dimension [C, range, angle]
    confmap = np.reshape(confmap_gt, (1, confmap_shape[0], 1, confmap_shape[1], confmap_shape[2]))
    confmap = torch.from_numpy(confmap)
    # new dimesinon [1, C, 1, range, angle]

    data_ra, data_va, confmap = transition_angle(data_ra, data_va, confmap, trans_angle=shift_angle)
    data_ra, data_rv, confmap = transition_range(data_ra, data_rv, confmap, trans_range=shift_range)

    # retrive dimension
    RA_slice = np.transpose(np.squeeze(data_ra.cpu().detach().numpy()),
                            (2, 3, 1, 0))  # new dimesinon [range, angle, chirps, C]
    RV_slice = np.transpose(np.squeeze(data_rv.cpu().detach().numpy()),
                            (1, 2, 0))  # new dimesinon [range, velocity, 2chirps]
    VA_slice = np.transpose(np.squeeze(data_va.cpu().detach().numpy()),
                            (1, 2, 0))  # new dimesinon [angle, velocity, 2chirps]
    confmap_gt = np.squeeze(confmap.cpu().detach().numpy())  # new dimesinon [C, range, angle]

    return RA_slice, RV_slice, VA_slice, confmap_gt


def calculate_obj_velocity(RA_slice, rng_idx, agl_idx):
    # RA slice [range, angle, chirps, real/imag]
    ra_cmplx = RA_slice[:, :, :, 0] + RA_slice[:, :, :, 1] * 1j
    chirp_data = ra_cmplx[rng_idx - 1:rng_idx + 2, agl_idx - 1:agl_idx + 2, :]
    rav_data = np.abs(velocity_fft(chirp_data))  # 256 points velocity fft
    _, _, dop_idx = np.unravel_index(rav_data.argmax(), rav_data.shape)  # maximum dop idx is 256

    return dop_idx


def generate_mix_ped(type='vertical'):
    # 2019_05_29_pm3s000: frame all
    root_dir = '/media/admin-cmmb/Elements/CRdataset/'
    root_dir_store = '/mnt/sda/3DRadardata/'
    root_imag_dir = '/mnt/nas_crdataset/'
    data_dir = './data/'
    confmap_dir = os.path.join(data_dir, 'confmaps_gt')
    detail_dir = os.path.join(data_dir, 'data_details')
    set_type = 'train_new'
    # seqs = ['2019_05_29_pm3s000', '2019_04_30_pm2s003', '2019_04_09_pms2000', '2019_05_23_pm2s011']
    # seqs = ['2019_05_29_pm3s000', '2019_04_30_pm2s003']
    seqs = ['2019_05_23_pm2s011']
    save_capture_date = '2020_00_00'
    save_seq_idx = 3

    for idx, seq in enumerate(seqs):
        capture_date = seq[0:10]
        save_seq = '2020_00_00_pmms' + str(save_seq_idx).zfill(3) # TODO: update save_seq_index
        seq_dir = os.path.join(root_dir, capture_date, seq, 'WIN_R_MAT')
        seq_path = os.path.join(root_imag_dir, capture_date, seq)
        seq_image_dir = os.path.join(root_imag_dir, capture_date, seq, 'images')

        ra_frame_offset = 0
        detail_list = [[], ra_frame_offset]
        confmap_list = [[], []]

        save_dir_rv = os.path.join(root_dir_store, save_capture_date, save_seq, 'RV_NPY')
        save_dir_ra = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', '0000')
        save_dir_ra2 = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', '0128')
        save_dir_va = os.path.join(root_dir_store, save_capture_date, save_seq, 'VA_NPY')
        save_dir_image = os.path.join(root_dir_store, save_capture_date, save_seq, 'images')

        if not os.path.exists(save_dir_rv):
            os.makedirs(save_dir_rv)
        if not os.path.exists(save_dir_ra):
            os.makedirs(save_dir_ra)
        if not os.path.exists(save_dir_ra2):
            os.makedirs(save_dir_ra2)
        if not os.path.exists(save_dir_va):
            os.makedirs(save_dir_va)
        if not os.path.exists(save_dir_image):
            os.makedirs(save_dir_image)
        files = sorted(os.listdir(seq_dir))
        images = sorted(os.listdir(seq_image_dir))
        print('Processing ', seq)

        # read label
        try:
            obj_info_list = read_ra_labels_csv(seq_path)
        except Exception as e:
            print("Load sequence %s failed!" % seq_path)
            continue

        if len(obj_info_list) <= len(files):
            files = files[len(files) - len(obj_info_list):]
        else:
            raise ValueError

        for idf, file in enumerate(files):
            file_dir = os.path.join(seq_dir, file)
            imag_dir = os.path.join(seq_image_dir, images[idf])

            mat = spio.loadmat(file_dir, squeeze_me=True)
            data = np.asarray(mat["R_data"])
            RA_slice = produce_RA_slice(data)
            RV_slice, rv_raw1, rv_raw2 = produce_RV_slice(data)
            VA_slice = produce_VA_slice(rv_raw1, rv_raw2)
            # generate confidence map
            obj_info = obj_info_list[idf]
            confmap_gt = np.zeros((n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                                  dtype=float)
            confmap_gt[-1, :, :] = 1.0
            if len(obj_info) != 0:
                confmap_gt = generate_confmap(obj_info)
                confmap_gt = normalize_confmap(confmap_gt)
                confmap_gt = add_noise_channel(confmap_gt)
            assert confmap_gt.shape == (n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

            ct_data = data - np.mean(data, axis=2, keepdims=True)  # remove static components
            ct_RV_slice, rv_raw1, rv_raw2 = produce_RV_slice(ct_data)
            ct_VA_slice = produce_VA_slice(rv_raw1, rv_raw2)
            ct_RA_slice = produce_RA_slice(ct_data)
            ct_confmap_gt = confmap_gt

            if idf % 60 == 0:
                chose_chirp = randint(0, 127)
                chose_chirp2 = chose_chirp + 127
                # choose two pedestrians that have similar velocities
                if len(obj_info) < 2:
                    # do nothing
                    ct_center_rng = 0
                    ct_center_agl = 0
                    bg_center_rng = 0
                    bg_center_agl = 0
                elif len(obj_info) == 2:
                    # choose the close one as the background (non moving)
                    if obj_info[0][0] <= obj_info[1][0]:
                        ct_center_rng = obj_info[1][0]
                        ct_center_agl = obj_info[1][1]
                        bg_center_rng = obj_info[0][0]
                        bg_center_agl = obj_info[0][1]
                    else:
                        ct_center_rng = obj_info[0][0]
                        ct_center_agl = obj_info[0][1]
                        bg_center_rng = obj_info[1][0]
                        bg_center_agl = obj_info[1][1]
                else:
                    obj_vel_info = []
                    for obj in obj_info:
                        obj_rng = obj[0]
                        obj_agl = obj[1]
                        # calculate the velocity of each object
                        obj_dop = calculate_obj_velocity(RA_slice, obj_rng, obj_agl)
                        obj_vel_info.append(obj_dop)
                    # choose two objects have similar velocity
                    minvd = 100
                    for idv, v in enumerate(obj_vel_info):
                        for idv2, v2 in enumerate(obj_vel_info):
                            if idv != idv2:
                                if abs(v - v2) <= minvd:
                                    minvd = abs(v - v2)
                                    min_idv = idv
                                    min_idv2 = idv2
                    # choose the close one as the background (non moving)
                    if obj_info[min_idv][0] <= obj_info[min_idv2][0]:
                        ct_center_rng = obj_info[min_idv2][0]
                        ct_center_agl = obj_info[min_idv2][1]
                        bg_center_rng = obj_info[min_idv][0]
                        bg_center_agl = obj_info[min_idv][1]
                    else:
                        ct_center_rng = obj_info[min_idv][0]
                        ct_center_agl = obj_info[min_idv][1]
                        bg_center_rng = obj_info[min_idv2][0]
                        bg_center_agl = obj_info[min_idv2][1]
                # move the content object to the backgound (only do range shift + angle shift)
                diff_range = bg_center_rng - ct_center_rng
                diff_angle = bg_center_agl - ct_center_agl
                if type == 'vertical':
                    shift_range = randint(diff_range+1, diff_range+7)
                    shift_angle = randint(diff_angle-4, diff_angle+4)
                elif type == 'horizontal':
                    shift_range = randint(diff_range-2, diff_range+2)
                    shift_angle = randint(diff_angle-4, diff_angle+4)

            ct_RA_slice, ct_RV_slice, ct_VA_slice, ct_confmap_gt = \
                translation(ct_RA_slice, ct_RV_slice, ct_VA_slice, ct_confmap_gt, shift_range, shift_angle)

            RA_slice = RA_slice + ct_RA_slice
            RV_slice = RV_slice + ct_RV_slice
            VA_slice = VA_slice + ct_VA_slice
            confmap_gt = np.maximum(confmap_gt, ct_confmap_gt)

            # save data
            new_file_name = str(idf).zfill(6) + '.npy'
            new_imag_name = str(idf).zfill(10) + '.jpg'
            save_file_name_rv = save_dir_rv + '/' + new_file_name
            save_file_name_va = save_dir_va + '/' + new_file_name
            save_file_name_ra = save_dir_ra + '/' + new_file_name
            save_file_name_ra2 = save_dir_ra2 + '/' + new_file_name
            save_file_name_imag = save_dir_image + '/' + new_imag_name
            # copy image and rename
            shutil.copyfile(imag_dir, save_file_name_imag)
            np.save(save_file_name_rv, RV_slice)
            np.save(save_file_name_va, VA_slice)
            np.save(save_file_name_ra, RA_slice[:, :, chose_chirp, :])
            np.save(save_file_name_ra2, RA_slice[:, :, chose_chirp2, :])

            # prepare files
            path = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', "%04d", "%06d.npy")
            detail_list[0].append(path)
            confmap_list[0].append(confmap_gt)
            confmap_list[1].append(obj_info)
            print('finished ', file)

        save_seq_idx += 1
        confmap_list[0] = np.array(confmap_list[0])
        # save pkl files
        pickle.dump(confmap_list, open(os.path.join(confmap_dir, set_type, save_seq + '.pkl'), 'wb'))
        # save pkl files
        pickle.dump(detail_list, open(os.path.join(detail_dir, set_type, save_seq + '.pkl'), 'wb'))


def generate_mix_staticped(type='vertical'):
    # 2019_05_29_pm3s000: frame all
    root_dir = '/media/admin-cmmb/Elements/CRdataset/'
    root_dir_store = '/mnt/sda/3DRadardata/'
    root_imag_dir = '/mnt/nas_crdataset/'
    data_dir = './data/'
    confmap_dir = os.path.join(data_dir, 'confmaps_gt')
    detail_dir = os.path.join(data_dir, 'data_details')
    set_type = 'train_new'
    seqs = ['2019_04_09_pss1003', '2019_04_30_ps1s005', '2019_05_29_ps1s006']
    save_capture_date = '2020_00_00'
    save_seq_idx = 0

    for idx, seq in enumerate(seqs):
        capture_date = seq[0:10]
        save_seq = '2020_00_00_psms' + str(save_seq_idx).zfill(3) # TODO: update save_seq_index
        seq_dir = os.path.join(root_dir, capture_date, seq, 'WIN_R_MAT')
        seq_path = os.path.join(root_imag_dir, capture_date, seq)
        seq_image_dir = os.path.join(root_imag_dir, capture_date, seq, 'images')

        ra_frame_offset = 0
        detail_list = [[], ra_frame_offset]
        confmap_list = [[], []]

        save_dir_rv = os.path.join(root_dir_store, save_capture_date, save_seq, 'RV_NPY')
        save_dir_ra = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', '0000')
        save_dir_ra2 = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', '0128')
        save_dir_va = os.path.join(root_dir_store, save_capture_date, save_seq, 'VA_NPY')
        save_dir_image = os.path.join(root_dir_store, save_capture_date, save_seq, 'images')

        if not os.path.exists(save_dir_rv):
            os.makedirs(save_dir_rv)
        if not os.path.exists(save_dir_ra):
            os.makedirs(save_dir_ra)
        if not os.path.exists(save_dir_ra2):
            os.makedirs(save_dir_ra2)
        if not os.path.exists(save_dir_va):
            os.makedirs(save_dir_va)
        if not os.path.exists(save_dir_image):
            os.makedirs(save_dir_image)
        files = sorted(os.listdir(seq_dir))
        images = sorted(os.listdir(seq_image_dir))
        print('Processing ', seq)

        # read label
        try:
            obj_info_list = read_ra_labels_csv(seq_path)
        except Exception as e:
            print("Load sequence %s failed!" % seq_path)
            continue

        if len(obj_info_list) <= len(files):
            files = files[len(files) - len(obj_info_list):]
        else:
            raise ValueError

        for idf, file in enumerate(files[0:300]):
            file_dir = os.path.join(seq_dir, file)
            imag_dir = os.path.join(seq_image_dir, images[idf])

            mat = spio.loadmat(file_dir, squeeze_me=True)
            data = np.asarray(mat["R_data"])
            RA_slice = produce_RA_slice(data)
            RV_slice, rv_raw1, rv_raw2 = produce_RV_slice(data)
            VA_slice = produce_VA_slice(rv_raw1, rv_raw2)
            # generate confidence map
            obj_info = obj_info_list[idf]
            confmap_gt = np.zeros((n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                                  dtype=float)
            confmap_gt[-1, :, :] = 1.0
            if len(obj_info) != 0:
                confmap_gt = generate_confmap(obj_info)
                confmap_gt = normalize_confmap(confmap_gt)
                confmap_gt = add_noise_channel(confmap_gt)
            assert confmap_gt.shape == (n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])


            if idf % 30 == 0:
                chose_chirp = randint(0, 127)
                chose_chirp2 = chose_chirp + 127
                # choose two pedestrians that have similar velocities
                if len(obj_info) == 0:
                    # do nothing
                    shift_range = 0
                    shift_angle = 0
                else:
                    shift_range = randint(-2, 2)
                    shift_angle = randint(-2, 2)

            if shift_angle != 0 and shift_range != 0:
                ct_RA_slice, ct_RV_slice, ct_VA_slice, ct_confmap_gt = \
                    translation(RA_slice, RV_slice, VA_slice, confmap_gt, shift_range, shift_angle)

            RA_slice = RA_slice + ct_RA_slice
            RV_slice = RV_slice + ct_RV_slice
            VA_slice = VA_slice + ct_VA_slice
            confmap_gt = np.maximum(confmap_gt, ct_confmap_gt)

            # save data
            new_file_name = str(idf).zfill(6) + '.npy'
            new_imag_name = str(idf).zfill(10) + '.jpg'
            save_file_name_rv = save_dir_rv + '/' + new_file_name
            save_file_name_va = save_dir_va + '/' + new_file_name
            save_file_name_ra = save_dir_ra + '/' + new_file_name
            save_file_name_ra2 = save_dir_ra2 + '/' + new_file_name
            save_file_name_imag = save_dir_image + '/' + new_imag_name
            # copy image and rename
            shutil.copyfile(imag_dir, save_file_name_imag)
            np.save(save_file_name_rv, RV_slice)
            np.save(save_file_name_va, VA_slice)
            np.save(save_file_name_ra, RA_slice[:, :, chose_chirp, :])
            np.save(save_file_name_ra2, RA_slice[:, :, chose_chirp2, :])

            # prepare files
            path = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', "%04d", "%06d.npy")
            detail_list[0].append(path)
            confmap_list[0].append(confmap_gt)
            confmap_list[1].append(obj_info)
            print('finished ', file)

        save_seq_idx += 1
        confmap_list[0] = np.array(confmap_list[0])
        # save pkl files
        pickle.dump(confmap_list, open(os.path.join(confmap_dir, set_type, save_seq + '.pkl'), 'wb'))
        # save pkl files
        pickle.dump(detail_list, open(os.path.join(detail_dir, set_type, save_seq + '.pkl'), 'wb'))


def generate_mix_ped_cyc():
    # 2019_04_30_pcms001: frame all
    # 2019_04_30_cm1s000: frame 210~300
    # 2019_05_29_cm1s014: frame 40-180, 640-690
    # 2019_05_29_cm1s015: frame 80~430
    # 2019_05_29_cm1s016: frame 170~270
    # 2019_05_29_cm1s017: frame 335-500

    # content
    root_dir = '/media/admin-cmmb/Elements/CRdataset/'
    root_dir_store = '/mnt/sda/3DRadardata/'
    root_imag_dir = '/mnt/nas_crdataset/'
    data_dir = './data/'
    confmap_dir = os.path.join(data_dir, 'confmaps_gt')
    detail_dir = os.path.join(data_dir, 'data_details')
    set_type = 'train_new'
    # bg_seqs = [['2019_04_30_pcms001'], ['2019_04_30_cm1s000', '2019_05_29_cm1s014', '2019_05_29_cm1s015', \
    #                                  '2019_05_29_cm1s016', '2019_05_29_cm1s017']]
    bg_seqs = ['2019_04_09_bms1001', '2019_05_29_bcms000']
    ct_seqs = ['2019_05_23_pm1s012', '2019_05_29_pcms005']
    save_seq_idx = 0  # TODO: change it accroding to scenario
    save_capture_date = '2020_00_00'

    for idx, seq in enumerate(bg_seqs):
        for idy, ct_seq in enumerate(ct_seqs):
            # read background data files
            capture_date = seq[0:10]
            seq_dir = os.path.join(root_dir, capture_date, seq, 'WIN_R_MAT')
            seq_path = os.path.join(root_imag_dir, capture_date, seq)
            seq_image_dir = os.path.join(root_imag_dir, capture_date, seq, 'images')

            files = sorted(os.listdir(seq_dir))
            images = sorted(os.listdir(seq_image_dir))
            print('Processing ', seq)
            # use labelled RAMap and prepare ground truth
            try:
                obj_info_list = read_ra_labels_csv(seq_path)
            except Exception as e:
                print("Load sequence %s failed!" % seq_path)
                print(e)
                continue

            # read content data files
            ct_capture_date = ct_seq[0:10]
            ct_seq_dir = os.path.join(root_dir, ct_capture_date, ct_seq, 'WIN_R_MAT')
            ct_files = sorted(os.listdir(ct_seq_dir))
            # read RA labels
            ct_seq_path = os.path.join(root_imag_dir, ct_capture_date, ct_seq)
            try:
                ct_obj_info_list = read_ra_labels_csv(ct_seq_path)
            except Exception as e:
                print("Load sequence %s failed!" % seq_path)
                print(e)
                continue

            if len(ct_obj_info_list) <= len(ct_files):
                ct_files = ct_files[len(ct_files) - len(ct_obj_info_list):]
            else:
                raise ValueError

            save_frame_idx = 0
            ra_frame_offset = 0
            detail_list = [[], ra_frame_offset]
            confmap_list = [[], []]
            shift = []

            save_seq = save_capture_date + '_pbms' + str(save_seq_idx).zfill(3)
            save_dir_rv = os.path.join(root_dir_store, save_capture_date, save_seq, 'RV_NPY')
            save_dir_ra = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', '0000')
            save_dir_ra2 = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', '0128')
            save_dir_va = os.path.join(root_dir_store, save_capture_date, save_seq, 'VA_NPY')
            save_dir_image = os.path.join(root_dir_store, save_capture_date, save_seq, 'images')
            store_info_dir = os.path.join(root_dir_store, save_capture_date, save_seq) + '/seq_info.txt'
            if not os.path.exists(save_dir_rv):
                os.makedirs(save_dir_rv)
            if not os.path.exists(save_dir_ra):
                os.makedirs(save_dir_ra)
            if not os.path.exists(save_dir_ra2):
                os.makedirs(save_dir_ra2)
            if not os.path.exists(save_dir_va):
                os.makedirs(save_dir_va)
            if not os.path.exists(save_dir_image):
                os.makedirs(save_dir_image)
            print('Making sequence: ', save_seq)

            for idf, file in enumerate(files):
                file_dir = os.path.join(seq_dir, file)
                imag_dir = os.path.join(seq_image_dir, images[idf])
                mat = spio.loadmat(file_dir, squeeze_me=True)
                data = np.asarray(mat["R_data"])
                RV_slice, rv_raw1, rv_raw2 = produce_RV_slice(data)
                VA_slice = produce_VA_slice(rv_raw1, rv_raw2)
                RA_slice = produce_RA_slice(data)
                confmap_gt = np.zeros((n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                                      dtype=float)
                confmap_gt[-1, :, :] = 1.0
                obj_info = obj_info_list[idf]
                if len(obj_info) != 0:
                    confmap_gt = generate_confmap(obj_info)
                    confmap_gt = normalize_confmap(confmap_gt)
                    confmap_gt = add_noise_channel(confmap_gt)
                assert confmap_gt.shape == (
                    n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

                ct_file_dir = os.path.join(ct_seq_dir, ct_files[save_frame_idx % len(ct_files)])
                ct_mat = spio.loadmat(ct_file_dir, squeeze_me=True)  # TODO: update save_frame_idx
                ct_data = np.asarray(ct_mat["R_data"])
                ct_data = ct_data - np.mean(ct_data, axis=2, keepdims=True)  # remove static components
                ct_RV_slice, rv_raw1, rv_raw2 = produce_RV_slice(ct_data)
                ct_VA_slice = produce_VA_slice(rv_raw1, rv_raw2)
                ct_RA_slice = produce_RA_slice(ct_data)

                ct_confmap_gt = np.zeros(
                    (n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                    dtype=float)
                ct_confmap_gt[-1, :, :] = 1.0
                ct_obj_info = ct_obj_info_list[save_frame_idx % len(ct_files)]
                if len(ct_obj_info) != 0:
                    ct_confmap_gt = generate_confmap(ct_obj_info)
                    ct_confmap_gt = normalize_confmap(ct_confmap_gt)
                    ct_confmap_gt = add_noise_channel(ct_confmap_gt)
                assert ct_confmap_gt.shape == (
                    n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

                # merge two RA slices: remove the static frequerncy in content and then add it to background
                # translate the content data
                if save_frame_idx % 30 == 0:
                    chose_chirp = randint(0, 127)
                    chose_chirp2 = chose_chirp + 127
                    # find a car in obj_info
                    need_info_cyc = find_obj_info('cyclist', obj_info)
                    need_info_ped = find_obj_info('pedestrian', ct_obj_info)

                    if need_info_ped is not None and need_info_cyc is not None:
                        diff_range = need_info_cyc[0] - need_info_ped[0]
                        diff_angle = need_info_cyc[1] - need_info_ped[1]
                        shift_range = randint(diff_range - 2, diff_range + 2)
                        shift_angle = randint(diff_angle - 3, diff_angle + 3)
                    else:
                        shift_angle = randint(-10, 10)
                        shift_range = randint(-10, 10)

                    shift.append([shift_range, shift_angle])

                ct_RA_slice, ct_RV_slice, ct_VA_slice, ct_confmap_gt = \
                    translation(ct_RA_slice, ct_RV_slice, ct_VA_slice, ct_confmap_gt, shift_range, shift_angle)
                # print(type(RA_slice), type(RV_slice), type(VA_slice), type(confmap_gt))
                # print(RA_slice.shape, RV_slice.shape, VA_slice.shape, confmap_gt.shape)
                # input()

                RA_slice = RA_slice + ct_RA_slice
                RV_slice = RV_slice + ct_RV_slice
                VA_slice = VA_slice + ct_VA_slice
                confmap_gt = np.maximum(confmap_gt, ct_confmap_gt)
                obj_info = obj_info + ct_obj_info

                # save data
                new_file_name = str(save_frame_idx).zfill(6) + '.npy'
                new_imag_name = str(save_frame_idx).zfill(10) + '.jpg'
                save_file_name_rv = save_dir_rv + '/' + new_file_name
                save_file_name_va = save_dir_va + '/' + new_file_name
                save_file_name_ra = save_dir_ra + '/' + new_file_name
                save_file_name_ra2 = save_dir_ra2 + '/' + new_file_name
                save_file_name_imag = save_dir_image + '/' + new_imag_name
                # copy image and rename
                shutil.copyfile(imag_dir, save_file_name_imag)
                np.save(save_file_name_rv, RV_slice)
                np.save(save_file_name_va, VA_slice)
                np.save(save_file_name_ra, RA_slice[:, :, chose_chirp, :])
                np.save(save_file_name_ra2, RA_slice[:, :, chose_chirp2, :])
                # prepare files
                path = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', "%04d", "%06d.npy")
                detail_list[0].append(path)
                confmap_list[0].append(confmap_gt)
                confmap_list[1].append(obj_info)
                print('finished ', file)
                save_frame_idx += 1

            confmap_list[0] = np.array(confmap_list[0])
            # save pkl files
            pickle.dump(confmap_list, open(os.path.join(confmap_dir, set_type, save_seq + '.pkl'), 'wb'))
            # save pkl files
            pickle.dump(detail_list, open(os.path.join(detail_dir, set_type, save_seq + '.pkl'), 'wb'))
            save_seq_idx += 1

            # write to file
            with open(store_info_dir, 'w+') as filehandle:
                filehandle.write('content data: %s background data: %s\n' % (ct_seq, seq))
                for cid, listitem in enumerate(shift):
                    # frame_id, range_shift, angle_shift
                    filehandle.write('%d %d\n' % (listitem[0], listitem[1]))


def generate_mix_car_pedcyc():
    # 2019_04_30_pcms001: frame all
    # 2019_04_30_cm1s000: frame 210~300
    # 2019_05_29_cm1s014: frame 40-180, 640-690
    # 2019_05_29_cm1s015: frame 80~430
    # 2019_05_29_cm1s016: frame 170~270
    # 2019_05_29_cm1s017: frame 335-500

    # content

    root_dir = '/media/admin-cmmb/Elements/CRdataset/'
    root_dir_store = '/mnt/sda/3DRadardata/'
    root_imag_dir = '/mnt/nas_crdataset/'
    data_dir = './data/'
    confmap_dir = os.path.join(data_dir, 'confmaps_gt')
    detail_dir = os.path.join(data_dir, 'data_details')
    set_type = 'train_new'
    bg_seqs = [['2019_04_30_pcms001'], ['2019_04_30_cm1s000', '2019_05_29_cm1s014', '2019_05_29_cm1s015', \
                                     '2019_05_29_cm1s016', '2019_05_29_cm1s017']]
    # bg_seqs = [['2019_04_30_cm1s000', '2019_05_29_cm1s014', '2019_05_29_cm1s015', \
    #                                      '2019_05_29_cm1s016', '2019_05_29_cm1s017']]
    # bg_seqs = [['2019_04_30_pcms001']]
    # bg_seqs.reverse()
    useful_frames = [None, [[[210, 300]], [[40, 180], [640, 690]], [[80, 430]], [[170, 270]], [[335, 500]]]]
    # useful_frames = [None]
    # useful_frames = [[[[210, 300]], [[40, 180], [640, 690]], [[80, 430]], [[170, 270]], [[335, 500]]]]
    ct_seqs = ['2019_05_23_pm2s011', '2019_04_30_pbms002', '2019_04_09_bms1001']
    save_seq_idx = 0  # TODO: change it accroding to scenario
    save_capture_date = '2020_00_00'

    for idbs, big_seq in enumerate(bg_seqs):
        for idy, ct_seq in enumerate(ct_seqs):
            # read content data files
            ct_capture_date = ct_seq[0:10]
            ct_seq_dir = os.path.join(root_dir, ct_capture_date, ct_seq, 'WIN_R_MAT')
            ct_files = sorted(os.listdir(ct_seq_dir))
            # read RA labels
            ct_seq_path = os.path.join(root_imag_dir, ct_capture_date, ct_seq)
            try:
                ct_obj_info_list = read_ra_labels_csv(ct_seq_path)
            except Exception as e:
                print("Load sequence %s failed!" % seq_path)
                print(e)
                continue
            if len(ct_obj_info_list) <= len(ct_files):
                ct_files = ct_files[len(ct_files)-len(ct_obj_info_list):]
            else:
                raise ValueError

            biguseful_frame = useful_frames[idbs]
            save_frame_idx = 0
            ra_frame_offset = 0
            detail_list = [[], ra_frame_offset]
            confmap_list = [[], []]
            shift = []

            save_seq = save_capture_date + '_cmms' + str(save_seq_idx).zfill(3)
            save_dir_rv = os.path.join(root_dir_store, save_capture_date, save_seq, 'RV_NPY')
            save_dir_ra = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', '0000')
            save_dir_ra2 = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', '0128')
            save_dir_va = os.path.join(root_dir_store, save_capture_date, save_seq, 'VA_NPY')
            save_dir_image = os.path.join(root_dir_store, save_capture_date, save_seq, 'images')
            store_info_dir = os.path.join(root_dir_store, save_capture_date, save_seq) + '/seq_info.txt'
            if not os.path.exists(save_dir_rv):
                os.makedirs(save_dir_rv)
            if not os.path.exists(save_dir_ra):
                os.makedirs(save_dir_ra)
            if not os.path.exists(save_dir_ra2):
                os.makedirs(save_dir_ra2)
            if not os.path.exists(save_dir_va):
                os.makedirs(save_dir_va)
            if not os.path.exists(save_dir_image):
                os.makedirs(save_dir_image)

            print('Making sequence: ', save_seq)
            for idx, seq in enumerate(big_seq):
                capture_date = seq[0:10]
                seq_dir = os.path.join(root_dir, capture_date, seq, 'WIN_R_MAT')
                seq_path = os.path.join(root_imag_dir, capture_date, seq)
                seq_image_dir = os.path.join(root_imag_dir, capture_date, seq, 'images')

                files = sorted(os.listdir(seq_dir))
                images = sorted(os.listdir(seq_image_dir))
                print('Processing ', seq)
                if biguseful_frame is not None:
                    useful_frame = biguseful_frame[idx]
                    for segm in useful_frame:
                        start_frame = segm[0]
                        end_frame = segm[1]
                        print('Assemble frames:', start_frame, ' ~ ', end_frame, 'in this squence')
                        # process, save data and prepare data
                        # use labelled RAMap and prepare ground truth
                        try:
                            obj_info_list = read_ra_labels_csv(seq_path)
                        except Exception as e:
                            print("Load sequence %s failed!" % seq_path)
                            print(e)
                            continue

                        for idf, file in enumerate(files[start_frame:end_frame]):
                            file_dir = os.path.join(seq_dir, file)
                            imag_dir = os.path.join(seq_image_dir, images[idf + start_frame])
                            mat = spio.loadmat(file_dir, squeeze_me=True)
                            data = np.asarray(mat["R_data"])
                            RV_slice, rv_raw1, rv_raw2 = produce_RV_slice(data)
                            VA_slice = produce_VA_slice(rv_raw1, rv_raw2)
                            RA_slice = produce_RA_slice(data)
                            confmap_gt = np.zeros((n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                                                  dtype=float)
                            confmap_gt[-1, :, :] = 1.0
                            obj_info = obj_info_list[start_frame+idf]
                            if len(obj_info) != 0:
                                confmap_gt = generate_confmap(obj_info)
                                confmap_gt = normalize_confmap(confmap_gt)
                                confmap_gt = add_noise_channel(confmap_gt)
                            assert confmap_gt.shape == (
                                n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

                            ct_file_dir = os.path.join(ct_seq_dir, ct_files[save_frame_idx % len(ct_files)])
                            ct_mat = spio.loadmat(ct_file_dir, squeeze_me=True)  # TODO: update save_frame_idx
                            ct_data = np.asarray(ct_mat["R_data"])
                            ct_data = ct_data - np.mean(ct_data, axis=2, keepdims=True)  # remove static components
                            ct_RV_slice, rv_raw1, rv_raw2 = produce_RV_slice(ct_data)
                            ct_VA_slice = produce_VA_slice(rv_raw1, rv_raw2)
                            ct_RA_slice = produce_RA_slice(ct_data)

                            ct_confmap_gt = np.zeros(
                                (n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                                dtype=float)
                            ct_confmap_gt[-1, :, :] = 1.0
                            ct_obj_info = ct_obj_info_list[save_frame_idx % len(ct_files)]
                            if len(ct_obj_info) != 0:
                                ct_confmap_gt = generate_confmap(ct_obj_info)
                                ct_confmap_gt = normalize_confmap(ct_confmap_gt)
                                ct_confmap_gt = add_noise_channel(ct_confmap_gt)
                            assert ct_confmap_gt.shape == (
                                n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

                            # merge two RA slices: remove the static frequerncy in content and then add it to background
                            # translate the content data
                            if save_frame_idx % 100 == 0:
                                # random translate left and upword
                                shift_angle = randint(-10, 10)
                                shift_range = randint(-10, 10)
                                chose_chirp = randint(0, 127)
                                chose_chirp2 = chose_chirp + 127
                                shift.append([shift_range, shift_angle])

                            ct_RA_slice, ct_RV_slice, ct_VA_slice, ct_confmap_gt = \
                                translation(ct_RA_slice, ct_RV_slice, ct_VA_slice, ct_confmap_gt, shift_range, shift_angle)
                            # print(type(RA_slice), type(RV_slice), type(VA_slice), type(confmap_gt))
                            # print(RA_slice.shape, RV_slice.shape, VA_slice.shape, confmap_gt.shape)
                            # input()

                            RA_slice = RA_slice + ct_RA_slice
                            RV_slice = RV_slice + ct_RV_slice
                            VA_slice = VA_slice + ct_VA_slice
                            confmap_gt = np.maximum(confmap_gt, ct_confmap_gt)
                            obj_info = obj_info + ct_obj_info

                            # save data
                            new_file_name = str(save_frame_idx).zfill(6) + '.npy'
                            new_imag_name = str(save_frame_idx).zfill(10) + '.jpg'
                            save_file_name_rv = save_dir_rv + '/' + new_file_name
                            save_file_name_va = save_dir_va + '/' + new_file_name
                            save_file_name_ra = save_dir_ra + '/' + new_file_name
                            save_file_name_ra2 = save_dir_ra2 + '/' + new_file_name
                            save_file_name_imag = save_dir_image + '/' + new_imag_name
                            # copy image and rename
                            shutil.copyfile(imag_dir, save_file_name_imag)
                            np.save(save_file_name_rv, RV_slice)
                            np.save(save_file_name_va, VA_slice)
                            np.save(save_file_name_ra, RA_slice[:, :, chose_chirp, :])
                            np.save(save_file_name_ra2, RA_slice[:, :, chose_chirp2, :])
                            # prepare files
                            path = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', "%04d", "%06d.npy")
                            detail_list[0].append(path)
                            confmap_list[0].append(confmap_gt)
                            confmap_list[1].append(obj_info)
                            print('finished ', file)
                            save_frame_idx += 1

                else:
                    print('Assemble all frames in this squence')
                    # use labelled RAMap and prepare ground truth
                    try:
                        obj_info_list = read_ra_labels_csv(seq_path)
                    except Exception as e:
                        print("Load sequence %s failed!" % seq_path)
                        print(e)
                        continue

                    for idf, file in enumerate(files):
                        file_dir = os.path.join(seq_dir, file)
                        imag_dir = os.path.join(seq_image_dir, images[idf])
                        mat = spio.loadmat(file_dir, squeeze_me=True)
                        data = np.asarray(mat["R_data"])
                        RV_slice, rv_raw1, rv_raw2 = produce_RV_slice(data)
                        VA_slice = produce_VA_slice(rv_raw1, rv_raw2)
                        RA_slice = produce_RA_slice(data)
                        confmap_gt = np.zeros((n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                                              dtype=float)
                        confmap_gt[-1, :, :] = 1.0
                        obj_info = obj_info_list[idf]
                        if len(obj_info) != 0:
                            confmap_gt = generate_confmap(obj_info)
                            confmap_gt = normalize_confmap(confmap_gt)
                            confmap_gt = add_noise_channel(confmap_gt)
                        assert confmap_gt.shape == (
                            n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

                        ct_file_dir = os.path.join(ct_seq_dir, ct_files[save_frame_idx % len(ct_files)])
                        ct_mat = spio.loadmat(ct_file_dir, squeeze_me=True)  # TODO: update save_frame_idx
                        ct_data = np.asarray(ct_mat["R_data"])
                        ct_data = ct_data - np.mean(ct_data, axis=2, keepdims=True)  # remove static components
                        ct_RV_slice, rv_raw1, rv_raw2 = produce_RV_slice(ct_data)
                        ct_VA_slice = produce_VA_slice(rv_raw1, rv_raw2)
                        ct_RA_slice = produce_RA_slice(ct_data)

                        ct_confmap_gt = np.zeros(
                            (n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                            dtype=float)
                        ct_confmap_gt[-1, :, :] = 1.0
                        ct_obj_info = ct_obj_info_list[save_frame_idx % len(ct_files)]
                        if len(ct_obj_info) != 0:
                            ct_confmap_gt = generate_confmap(ct_obj_info)
                            ct_confmap_gt = normalize_confmap(ct_confmap_gt)
                            ct_confmap_gt = add_noise_channel(ct_confmap_gt)
                        assert ct_confmap_gt.shape == (
                            n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

                        # merge two RA slices: remove the static frequerncy in content and then add it to background
                        # translate the content data
                        if save_frame_idx % 100 == 0:
                            # random translate left and upword
                            shift_angle = randint(-10, 10)
                            shift_range = randint(-10, 10)
                            chose_chirp = randint(0, 127)
                            chose_chirp2 = chose_chirp + 127
                            shift.append([shift_range, shift_angle])

                        ct_RA_slice, ct_RV_slice, ct_VA_slice, ct_confmap_gt = \
                            translation(ct_RA_slice, ct_RV_slice, ct_VA_slice, ct_confmap_gt, shift_range, shift_angle)
                        # print(type(RA_slice), type(RV_slice), type(VA_slice), type(confmap_gt))
                        # print(RA_slice.shape, RV_slice.shape, VA_slice.shape, confmap_gt.shape)
                        # input()

                        RA_slice = RA_slice + ct_RA_slice
                        RV_slice = RV_slice + ct_RV_slice
                        VA_slice = VA_slice + ct_VA_slice
                        confmap_gt = np.maximum(confmap_gt, ct_confmap_gt)
                        obj_info = obj_info + ct_obj_info

                        # save data
                        new_file_name = str(save_frame_idx).zfill(6) + '.npy'
                        new_imag_name = str(save_frame_idx).zfill(10) + '.jpg'
                        save_file_name_rv = save_dir_rv + '/' + new_file_name
                        save_file_name_va = save_dir_va + '/' + new_file_name
                        save_file_name_ra = save_dir_ra + '/' + new_file_name
                        save_file_name_ra2 = save_dir_ra2 + '/' + new_file_name
                        save_file_name_imag = save_dir_image + '/' + new_imag_name
                        # copy image and rename
                        shutil.copyfile(imag_dir, save_file_name_imag)
                        np.save(save_file_name_rv, RV_slice)
                        np.save(save_file_name_va, VA_slice)
                        np.save(save_file_name_ra, RA_slice[:, :, chose_chirp, :])
                        np.save(save_file_name_ra2, RA_slice[:, :, chose_chirp2, :])
                        # prepare files
                        path = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', "%04d", "%06d.npy")
                        detail_list[0].append(path)
                        confmap_list[0].append(confmap_gt)
                        confmap_list[1].append(obj_info)
                        print('finished ', file)
                        save_frame_idx += 1

            confmap_list[0] = np.array(confmap_list[0])
            # save pkl files
            pickle.dump(confmap_list, open(os.path.join(confmap_dir, set_type, save_seq + '.pkl'), 'wb'))
            # save pkl files
            pickle.dump(detail_list, open(os.path.join(detail_dir, set_type, save_seq + '.pkl'), 'wb'))
            save_seq_idx += 1

            # write to file
            if len(big_seq) > 1:
                print_seq = ' '
                for seq in big_seq:
                    print_seq = print_seq + seq + ' '
            else:
                print_seq = big_seq[0]
            with open(store_info_dir, 'w+') as filehandle:
                filehandle.write('content data: %s background data: %s\n' % (ct_seq, print_seq))
                for cid, listitem in enumerate(shift):
                    # frame_id, range_shift, angle_shift
                    filehandle.write('%d %d\n' % (listitem[0], listitem[1]))


def generate_mix_staticcar_pedcyc():
    # 2019_04_30_pcms001: frame all
    # 2019_04_30_cm1s000: frame 210~300
    # 2019_05_29_cm1s014: frame 40-180, 640-690
    # 2019_05_29_cm1s015: frame 80~430
    # 2019_05_29_cm1s016: frame 170~270
    # 2019_05_29_cm1s017: frame 335-500

    # content

    root_dir = '/media/admin-cmmb/Elements/CRdataset/'
    root_dir_store = '/mnt/sda/3DRadardata/'
    root_imag_dir = '/mnt/nas_crdataset/'
    data_dir = './data/'
    confmap_dir = os.path.join(data_dir, 'confmaps_gt')
    detail_dir = os.path.join(data_dir, 'data_details')
    set_type = 'train_new'
    bg_seqs = [['2019_05_29_cs1s007'], ['2019_05_09_cs1s001']]
    # bg_seqs = [['2019_04_30_cm1s000', '2019_05_29_cm1s014', '2019_05_29_cm1s015', \
    #                                      '2019_05_29_cm1s016', '2019_05_29_cm1s017']]
    # bg_seqs = [['2019_04_30_pcms001']]
    # bg_seqs.reverse()
    useful_frames = [None, None]
    # useful_frames = [None]
    # useful_frames = [[[[210, 300]], [[40, 180], [640, 690]], [[80, 430]], [[170, 270]], [[335, 500]]]]
    ct_seqs = ['2019_05_23_pm1s013', '2019_05_23_pm1s014']
    save_seq_idx = 0  # TODO: change it accroding to scenario
    save_capture_date = '2020_00_00'

    for idbs, big_seq in enumerate(bg_seqs):
        for idy, ct_seq in enumerate(ct_seqs):
            # read content data files
            ct_capture_date = ct_seq[0:10]
            ct_seq_dir = os.path.join(root_dir, ct_capture_date, ct_seq, 'WIN_R_MAT')
            ct_files = sorted(os.listdir(ct_seq_dir))
            # read RA labels
            ct_seq_path = os.path.join(root_imag_dir, ct_capture_date, ct_seq)
            try:
                ct_obj_info_list = read_ra_labels_csv(ct_seq_path)
            except Exception as e:
                print("Load sequence %s failed!" % seq_path)
                print(e)
                continue
            if len(ct_obj_info_list) <= len(ct_files):
                ct_files = ct_files[len(ct_files)-len(ct_obj_info_list):]
            else:
                raise ValueError

            biguseful_frame = useful_frames[idbs]
            save_frame_idx = 0
            ra_frame_offset = 0
            detail_list = [[], ra_frame_offset]
            confmap_list = [[], []]
            shift = []

            save_seq = save_capture_date + '_csps' + str(save_seq_idx).zfill(3)
            save_dir_rv = os.path.join(root_dir_store, save_capture_date, save_seq, 'RV_NPY')
            save_dir_ra = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', '0000')
            save_dir_ra2 = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', '0128')
            save_dir_va = os.path.join(root_dir_store, save_capture_date, save_seq, 'VA_NPY')
            save_dir_image = os.path.join(root_dir_store, save_capture_date, save_seq, 'images')
            store_info_dir = os.path.join(root_dir_store, save_capture_date, save_seq) + '/seq_info.txt'
            if not os.path.exists(save_dir_rv):
                os.makedirs(save_dir_rv)
            if not os.path.exists(save_dir_ra):
                os.makedirs(save_dir_ra)
            if not os.path.exists(save_dir_ra2):
                os.makedirs(save_dir_ra2)
            if not os.path.exists(save_dir_va):
                os.makedirs(save_dir_va)
            if not os.path.exists(save_dir_image):
                os.makedirs(save_dir_image)

            print('Making sequence: ', save_seq)
            for idx, seq in enumerate(big_seq):
                capture_date = seq[0:10]
                seq_dir = os.path.join(root_dir, capture_date, seq, 'WIN_R_MAT')
                seq_path = os.path.join(root_imag_dir, capture_date, seq)
                seq_image_dir = os.path.join(root_imag_dir, capture_date, seq, 'images')

                files = sorted(os.listdir(seq_dir))
                images = sorted(os.listdir(seq_image_dir))
                print('Processing ', seq)
                if biguseful_frame is not None:
                    useful_frame = biguseful_frame[idx]
                    for segm in useful_frame:
                        start_frame = segm[0]
                        end_frame = segm[1]
                        print('Assemble frames:', start_frame, ' ~ ', end_frame, 'in this squence')
                        # process, save data and prepare data
                        # use labelled RAMap and prepare ground truth
                        try:
                            obj_info_list = read_ra_labels_csv(seq_path)
                        except Exception as e:
                            print("Load sequence %s failed!" % seq_path)
                            print(e)
                            continue

                        for idf, file in enumerate(files[start_frame:end_frame]):
                            file_dir = os.path.join(seq_dir, file)
                            imag_dir = os.path.join(seq_image_dir, images[idf + start_frame])
                            mat = spio.loadmat(file_dir, squeeze_me=True)
                            data = np.asarray(mat["R_data"])
                            RV_slice, rv_raw1, rv_raw2 = produce_RV_slice(data)
                            VA_slice = produce_VA_slice(rv_raw1, rv_raw2)
                            RA_slice = produce_RA_slice(data)
                            confmap_gt = np.zeros((n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                                                  dtype=float)
                            confmap_gt[-1, :, :] = 1.0
                            obj_info = obj_info_list[start_frame+idf]
                            if len(obj_info) != 0:
                                confmap_gt = generate_confmap(obj_info)
                                confmap_gt = normalize_confmap(confmap_gt)
                                confmap_gt = add_noise_channel(confmap_gt)
                            assert confmap_gt.shape == (
                                n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

                            ct_file_dir = os.path.join(ct_seq_dir, ct_files[save_frame_idx % len(ct_files)])
                            ct_mat = spio.loadmat(ct_file_dir, squeeze_me=True)  # TODO: update save_frame_idx
                            ct_data = np.asarray(ct_mat["R_data"])
                            ct_data = ct_data - np.mean(ct_data, axis=2, keepdims=True)  # remove static components
                            ct_RV_slice, rv_raw1, rv_raw2 = produce_RV_slice(ct_data)
                            ct_VA_slice = produce_VA_slice(rv_raw1, rv_raw2)
                            ct_RA_slice = produce_RA_slice(ct_data)

                            ct_confmap_gt = np.zeros(
                                (n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                                dtype=float)
                            ct_confmap_gt[-1, :, :] = 1.0
                            ct_obj_info = ct_obj_info_list[save_frame_idx % len(ct_files)]
                            if len(ct_obj_info) != 0:
                                ct_confmap_gt = generate_confmap(ct_obj_info)
                                ct_confmap_gt = normalize_confmap(ct_confmap_gt)
                                ct_confmap_gt = add_noise_channel(ct_confmap_gt)
                            assert ct_confmap_gt.shape == (
                                n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

                            # merge two RA slices: remove the static frequerncy in content and then add it to background
                            # translate the content data
                            if save_frame_idx % 100 == 0:
                                # random translate left and upword
                                shift_angle = randint(-10, 10)
                                shift_range = randint(-10, 10)
                                chose_chirp = randint(0, 127)
                                chose_chirp2 = chose_chirp + 127
                                shift.append([shift_range, shift_angle])

                            ct_RA_slice, ct_RV_slice, ct_VA_slice, ct_confmap_gt = \
                                translation(ct_RA_slice, ct_RV_slice, ct_VA_slice, ct_confmap_gt, shift_range, shift_angle)
                            # print(type(RA_slice), type(RV_slice), type(VA_slice), type(confmap_gt))
                            # print(RA_slice.shape, RV_slice.shape, VA_slice.shape, confmap_gt.shape)
                            # input()

                            RA_slice = RA_slice + ct_RA_slice
                            RV_slice = RV_slice + ct_RV_slice
                            VA_slice = VA_slice + ct_VA_slice
                            confmap_gt = np.maximum(confmap_gt, ct_confmap_gt)
                            obj_info = obj_info + ct_obj_info

                            # save data
                            new_file_name = str(save_frame_idx).zfill(6) + '.npy'
                            new_imag_name = str(save_frame_idx).zfill(10) + '.jpg'
                            save_file_name_rv = save_dir_rv + '/' + new_file_name
                            save_file_name_va = save_dir_va + '/' + new_file_name
                            save_file_name_ra = save_dir_ra + '/' + new_file_name
                            save_file_name_ra2 = save_dir_ra2 + '/' + new_file_name
                            save_file_name_imag = save_dir_image + '/' + new_imag_name
                            # copy image and rename
                            shutil.copyfile(imag_dir, save_file_name_imag)
                            np.save(save_file_name_rv, RV_slice)
                            np.save(save_file_name_va, VA_slice)
                            np.save(save_file_name_ra, RA_slice[:, :, chose_chirp, :])
                            np.save(save_file_name_ra2, RA_slice[:, :, chose_chirp2, :])
                            # prepare files
                            path = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', "%04d", "%06d.npy")
                            detail_list[0].append(path)
                            confmap_list[0].append(confmap_gt)
                            confmap_list[1].append(obj_info)
                            print('finished ', file)
                            save_frame_idx += 1

                else:
                    print('Assemble all frames in this squence')
                    # use labelled RAMap and prepare ground truth
                    try:
                        obj_info_list = read_ra_labels_csv(seq_path)
                    except Exception as e:
                        print("Load sequence %s failed!" % seq_path)
                        print(e)
                        continue

                    for idf, file in enumerate(files):
                        file_dir = os.path.join(seq_dir, file)
                        imag_dir = os.path.join(seq_image_dir, images[idf])
                        mat = spio.loadmat(file_dir, squeeze_me=True)
                        data = np.asarray(mat["R_data"])
                        RV_slice, rv_raw1, rv_raw2 = produce_RV_slice(data)
                        VA_slice = produce_VA_slice(rv_raw1, rv_raw2)
                        RA_slice = produce_RA_slice(data)
                        confmap_gt = np.zeros((n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                                              dtype=float)
                        confmap_gt[-1, :, :] = 1.0
                        obj_info = obj_info_list[idf]
                        if len(obj_info) != 0:
                            confmap_gt = generate_confmap(obj_info)
                            confmap_gt = normalize_confmap(confmap_gt)
                            confmap_gt = add_noise_channel(confmap_gt)
                        assert confmap_gt.shape == (
                            n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

                        ct_file_dir = os.path.join(ct_seq_dir, ct_files[save_frame_idx % len(ct_files)])
                        ct_mat = spio.loadmat(ct_file_dir, squeeze_me=True)  # TODO: update save_frame_idx
                        ct_data = np.asarray(ct_mat["R_data"])
                        ct_data = ct_data - np.mean(ct_data, axis=2, keepdims=True)  # remove static components
                        ct_RV_slice, rv_raw1, rv_raw2 = produce_RV_slice(ct_data)
                        ct_VA_slice = produce_VA_slice(rv_raw1, rv_raw2)
                        ct_RA_slice = produce_RA_slice(ct_data)

                        ct_confmap_gt = np.zeros(
                            (n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                            dtype=float)
                        ct_confmap_gt[-1, :, :] = 1.0
                        ct_obj_info = ct_obj_info_list[save_frame_idx % len(ct_files)]
                        if len(ct_obj_info) != 0:
                            ct_confmap_gt = generate_confmap(ct_obj_info)
                            ct_confmap_gt = normalize_confmap(ct_confmap_gt)
                            ct_confmap_gt = add_noise_channel(ct_confmap_gt)
                        assert ct_confmap_gt.shape == (
                            n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

                        # merge two RA slices: remove the static frequerncy in content and then add it to background
                        # translate the content data
                        if save_frame_idx % 100 == 0:
                            # random translate left and upword
                            shift_angle = randint(-10, 10)
                            shift_range = randint(-10, 10)
                            chose_chirp = randint(0, 127)
                            chose_chirp2 = chose_chirp + 127
                            shift.append([shift_range, shift_angle])

                        ct_RA_slice, ct_RV_slice, ct_VA_slice, ct_confmap_gt = \
                            translation(ct_RA_slice, ct_RV_slice, ct_VA_slice, ct_confmap_gt, shift_range, shift_angle)
                        # print(type(RA_slice), type(RV_slice), type(VA_slice), type(confmap_gt))
                        # print(RA_slice.shape, RV_slice.shape, VA_slice.shape, confmap_gt.shape)
                        # input()

                        RA_slice = RA_slice + ct_RA_slice
                        RV_slice = RV_slice + ct_RV_slice
                        VA_slice = VA_slice + ct_VA_slice
                        confmap_gt = np.maximum(confmap_gt, ct_confmap_gt)
                        obj_info = obj_info + ct_obj_info

                        # save data
                        new_file_name = str(save_frame_idx).zfill(6) + '.npy'
                        new_imag_name = str(save_frame_idx).zfill(10) + '.jpg'
                        save_file_name_rv = save_dir_rv + '/' + new_file_name
                        save_file_name_va = save_dir_va + '/' + new_file_name
                        save_file_name_ra = save_dir_ra + '/' + new_file_name
                        save_file_name_ra2 = save_dir_ra2 + '/' + new_file_name
                        save_file_name_imag = save_dir_image + '/' + new_imag_name
                        # copy image and rename
                        shutil.copyfile(imag_dir, save_file_name_imag)
                        np.save(save_file_name_rv, RV_slice)
                        np.save(save_file_name_va, VA_slice)
                        np.save(save_file_name_ra, RA_slice[:, :, chose_chirp, :])
                        np.save(save_file_name_ra2, RA_slice[:, :, chose_chirp2, :])
                        # prepare files
                        path = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', "%04d", "%06d.npy")
                        detail_list[0].append(path)
                        confmap_list[0].append(confmap_gt)
                        confmap_list[1].append(obj_info)
                        print('finished ', file)
                        save_frame_idx += 1

            confmap_list[0] = np.array(confmap_list[0])
            # save pkl files
            pickle.dump(confmap_list, open(os.path.join(confmap_dir, set_type, save_seq + '.pkl'), 'wb'))
            # save pkl files
            pickle.dump(detail_list, open(os.path.join(detail_dir, set_type, save_seq + '.pkl'), 'wb'))
            save_seq_idx += 1

            # write to file
            if len(big_seq) > 1:
                print_seq = ' '
                for seq in big_seq:
                    print_seq = print_seq + seq + ' '
            else:
                print_seq = big_seq[0]
            with open(store_info_dir, 'w+') as filehandle:
                filehandle.write('content data: %s background data: %s\n' % (ct_seq, print_seq))
                for cid, listitem in enumerate(shift):
                    # frame_id, range_shift, angle_shift
                    filehandle.write('%d %d\n' % (listitem[0], listitem[1]))


def generate_mixcross_car_pedcyc():
    # 2019_04_30_pcms001: frame all
    # 2019_04_30_cm1s000: frame 210~300
    # 2019_05_29_cm1s014: frame 40-180, 640-690
    # 2019_05_29_cm1s015: frame 80~430
    # 2019_05_29_cm1s016: frame 170~270
    # 2019_05_29_cm1s017: frame 335-500

    # content
    root_dir = '/media/admin-cmmb/Elements/CRdataset/'
    root_dir_store = '/mnt/sda/3DRadardata/'
    root_imag_dir = '/mnt/nas_crdataset/'
    data_dir = './data/'
    confmap_dir = os.path.join(data_dir, 'confmaps_gt')
    detail_dir = os.path.join(data_dir, 'data_details')
    set_type = 'train_new'
    # bg_seqs = [['2019_04_30_pcms001'], ['2019_04_30_cm1s000', '2019_05_29_cm1s014', '2019_05_29_cm1s015', \
    #                                  '2019_05_29_cm1s016', '2019_05_29_cm1s017']]
    bg_seqs = [['2019_04_30_cm1s000', '2019_05_29_cm1s014', '2019_05_29_cm1s015', \
                                         '2019_05_29_cm1s016', '2019_05_29_cm1s017']]
    # bg_seqs = [['2019_04_30_pcms001']]
    # bg_seqs.reverse()
    # useful_frames = [None, [[[210, 300]], [[40, 180], [640, 690]], [[80, 430]], [[170, 270]], [[335, 500]]]]
    # useful_frames = [None]
    useful_frames = [[[[210, 300]], [[40, 180], [640, 690]], [[80, 430]], [[170, 270]], [[335, 500]]]]
    ct_seqs = ['2019_05_23_pm2s011', '2019_04_30_pbms002', '2019_04_09_bms1001']
    save_seq_idx = 6  # TODO: change it accroding to scenario
    save_capture_date = '2020_00_00'

    for idbs, big_seq in enumerate(bg_seqs):
        for idy, ct_seq in enumerate(ct_seqs):
            # read content data files
            ct_capture_date = ct_seq[0:10]
            ct_seq_dir = os.path.join(root_dir, ct_capture_date, ct_seq, 'WIN_R_MAT')
            ct_files = sorted(os.listdir(ct_seq_dir))
            # read RA labels
            ct_seq_path = os.path.join(root_imag_dir, ct_capture_date, ct_seq)
            try:
                ct_obj_info_list = read_ra_labels_csv(ct_seq_path)
            except Exception as e:
                print("Load sequence %s failed!" % seq_path)
                print(e)
                continue
            if len(ct_obj_info_list) <= len(ct_files):
                ct_files = ct_files[len(ct_files)-len(ct_obj_info_list):]
            else:
                raise ValueError

            biguseful_frame = useful_frames[idbs]
            save_frame_idx = 0
            ra_frame_offset = 0
            detail_list = [[], ra_frame_offset]
            confmap_list = [[], []]
            shift = []

            save_seq = save_capture_date + '_cmms' + str(save_seq_idx).zfill(3)
            save_dir_rv = os.path.join(root_dir_store, save_capture_date, save_seq, 'RV_NPY')
            save_dir_ra = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', '0000')
            save_dir_ra2 = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', '0128')
            save_dir_va = os.path.join(root_dir_store, save_capture_date, save_seq, 'VA_NPY')
            save_dir_image = os.path.join(root_dir_store, save_capture_date, save_seq, 'images')
            store_info_dir = os.path.join(root_dir_store, save_capture_date, save_seq) + '/seq_info.txt'
            if not os.path.exists(save_dir_rv):
                os.makedirs(save_dir_rv)
            if not os.path.exists(save_dir_ra):
                os.makedirs(save_dir_ra)
            if not os.path.exists(save_dir_ra2):
                os.makedirs(save_dir_ra2)
            if not os.path.exists(save_dir_va):
                os.makedirs(save_dir_va)
            if not os.path.exists(save_dir_image):
                os.makedirs(save_dir_image)

            print('Making sequence: ', save_seq)
            for idx, seq in enumerate(big_seq):
                capture_date = seq[0:10]
                seq_dir = os.path.join(root_dir, capture_date, seq, 'WIN_R_MAT')
                seq_path = os.path.join(root_imag_dir, capture_date, seq)
                seq_image_dir = os.path.join(root_imag_dir, capture_date, seq, 'images')

                files = sorted(os.listdir(seq_dir))
                images = sorted(os.listdir(seq_image_dir))
                print('Processing ', seq)
                if biguseful_frame is not None:
                    useful_frame = biguseful_frame[idx]
                    for segm in useful_frame:
                        start_frame = segm[0]
                        end_frame = segm[1]
                        print('Assemble frames:', start_frame, ' ~ ', end_frame, 'in this squence')
                        # process, save data and prepare data
                        # use labelled RAMap and prepare ground truth
                        try:
                            obj_info_list = read_ra_labels_csv(seq_path)
                        except Exception as e:
                            print("Load sequence %s failed!" % seq_path)
                            print(e)
                            continue

                        for idf, file in enumerate(files[start_frame:end_frame]):
                            file_dir = os.path.join(seq_dir, file)
                            imag_dir = os.path.join(seq_image_dir, images[idf + start_frame])
                            mat = spio.loadmat(file_dir, squeeze_me=True)
                            data = np.asarray(mat["R_data"])
                            RV_slice, rv_raw1, rv_raw2 = produce_RV_slice(data)
                            VA_slice = produce_VA_slice(rv_raw1, rv_raw2)
                            RA_slice = produce_RA_slice(data)
                            confmap_gt = np.zeros((n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                                                  dtype=float)
                            confmap_gt[-1, :, :] = 1.0
                            obj_info = obj_info_list[start_frame+idf]
                            if len(obj_info) != 0:
                                confmap_gt = generate_confmap(obj_info)
                                confmap_gt = normalize_confmap(confmap_gt)
                                confmap_gt = add_noise_channel(confmap_gt)
                            assert confmap_gt.shape == (
                                n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

                            ct_file_dir = os.path.join(ct_seq_dir, ct_files[save_frame_idx % len(ct_files)])
                            ct_mat = spio.loadmat(ct_file_dir, squeeze_me=True)  # TODO: update save_frame_idx
                            ct_data = np.asarray(ct_mat["R_data"])
                            ct_data = ct_data - np.mean(ct_data, axis=2, keepdims=True)  # remove static components
                            ct_RV_slice, rv_raw1, rv_raw2 = produce_RV_slice(ct_data)
                            ct_VA_slice = produce_VA_slice(rv_raw1, rv_raw2)
                            ct_RA_slice = produce_RA_slice(ct_data)

                            ct_confmap_gt = np.zeros(
                                (n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                                dtype=float)
                            ct_confmap_gt[-1, :, :] = 1.0
                            ct_obj_info = ct_obj_info_list[save_frame_idx % len(ct_files)]
                            if len(ct_obj_info) != 0:
                                ct_confmap_gt = generate_confmap(ct_obj_info)
                                ct_confmap_gt = normalize_confmap(ct_confmap_gt)
                                ct_confmap_gt = add_noise_channel(ct_confmap_gt)
                            assert ct_confmap_gt.shape == (
                                n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

                            # merge two RA slices: remove the static frequerncy in content and then add it to background
                            # translate the content data
                            if save_frame_idx % 30 == 0:
                                chose_chirp = randint(0, 127)
                                chose_chirp2 = chose_chirp + 127
                                # find a car in obj_info
                                need_info_car = find_obj_info('car', obj_info)
                                need_info_ped = find_obj_info('pedestrian', ct_obj_info)
                                need_info_cyc = find_obj_info('cyclist', ct_obj_info)

                                if need_info_car is None:
                                    shift_angle = randint(-10, 10)
                                    shift_range = randint(-10, 10)
                                else:
                                    if need_info_ped is not None and need_info_cyc is not None:
                                        # random choose one
                                        if random() < 0.5:
                                            # choose ped
                                            need_info_ct = need_info_ped
                                        else:
                                            # choose cyclist
                                            need_info_ct = need_info_cyc
                                    elif need_info_ped is not None:
                                        need_info_ct = need_info_ped
                                    elif need_info_cyc is not None:
                                        need_info_ct = need_info_cyc
                                    else:
                                        need_info_ct = None

                                    if need_info_ct is None:
                                        shift_angle = randint(-10, 10)
                                        shift_range = randint(-10, 10)
                                    else:
                                        diff_range = need_info_car[0] - need_info_ct[0]
                                        diff_angle = need_info_car[1] - need_info_ct[1]
                                        if abs(diff_range) < abs(diff_angle):
                                            # only shift_range
                                            shift_range = randint(diff_range - 2, diff_range + 2)
                                            shift_angle = 0
                                        else:
                                            shift_range = 0
                                            shift_angle = randint(diff_angle - 3, diff_angle + 3)

                                shift.append([shift_range, shift_angle])

                            ct_RA_slice, ct_RV_slice, ct_VA_slice, ct_confmap_gt = \
                                translation(ct_RA_slice, ct_RV_slice, ct_VA_slice, ct_confmap_gt, shift_range, shift_angle)
                            # print(type(RA_slice), type(RV_slice), type(VA_slice), type(confmap_gt))
                            # print(RA_slice.shape, RV_slice.shape, VA_slice.shape, confmap_gt.shape)
                            # input()

                            RA_slice = RA_slice + ct_RA_slice
                            RV_slice = RV_slice + ct_RV_slice
                            VA_slice = VA_slice + ct_VA_slice
                            confmap_gt = np.maximum(confmap_gt, ct_confmap_gt)
                            obj_info = obj_info + ct_obj_info

                            # save data
                            new_file_name = str(save_frame_idx).zfill(6) + '.npy'
                            new_imag_name = str(save_frame_idx).zfill(10) + '.jpg'
                            save_file_name_rv = save_dir_rv + '/' + new_file_name
                            save_file_name_va = save_dir_va + '/' + new_file_name
                            save_file_name_ra = save_dir_ra + '/' + new_file_name
                            save_file_name_ra2 = save_dir_ra2 + '/' + new_file_name
                            save_file_name_imag = save_dir_image + '/' + new_imag_name
                            # copy image and rename
                            shutil.copyfile(imag_dir, save_file_name_imag)
                            np.save(save_file_name_rv, RV_slice)
                            np.save(save_file_name_va, VA_slice)
                            np.save(save_file_name_ra, RA_slice[:, :, chose_chirp, :])
                            np.save(save_file_name_ra2, RA_slice[:, :, chose_chirp2, :])
                            # prepare files
                            path = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', "%04d", "%06d.npy")
                            detail_list[0].append(path)
                            confmap_list[0].append(confmap_gt)
                            confmap_list[1].append(obj_info)
                            print('finished ', file)
                            save_frame_idx += 1

                else:
                    print('Assemble all frames in this squence')
                    # use labelled RAMap and prepare ground truth
                    try:
                        obj_info_list = read_ra_labels_csv(seq_path)
                    except Exception as e:
                        print("Load sequence %s failed!" % seq_path)
                        print(e)
                        continue

                    for idf, file in enumerate(files):
                        file_dir = os.path.join(seq_dir, file)
                        imag_dir = os.path.join(seq_image_dir, images[idf])
                        mat = spio.loadmat(file_dir, squeeze_me=True)
                        data = np.asarray(mat["R_data"])
                        RV_slice, rv_raw1, rv_raw2 = produce_RV_slice(data)
                        VA_slice = produce_VA_slice(rv_raw1, rv_raw2)
                        RA_slice = produce_RA_slice(data)
                        confmap_gt = np.zeros((n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                                              dtype=float)
                        confmap_gt[-1, :, :] = 1.0
                        obj_info = obj_info_list[idf]
                        if len(obj_info) != 0:
                            confmap_gt = generate_confmap(obj_info)
                            confmap_gt = normalize_confmap(confmap_gt)
                            confmap_gt = add_noise_channel(confmap_gt)
                        assert confmap_gt.shape == (
                            n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

                        ct_file_dir = os.path.join(ct_seq_dir, ct_files[save_frame_idx % len(ct_files)])
                        ct_mat = spio.loadmat(ct_file_dir, squeeze_me=True)  # TODO: update save_frame_idx
                        ct_data = np.asarray(ct_mat["R_data"])
                        ct_data = ct_data - np.mean(ct_data, axis=2, keepdims=True)  # remove static components
                        ct_RV_slice, rv_raw1, rv_raw2 = produce_RV_slice(ct_data)
                        ct_VA_slice = produce_VA_slice(rv_raw1, rv_raw2)
                        ct_RA_slice = produce_RA_slice(ct_data)

                        ct_confmap_gt = np.zeros(
                            (n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                            dtype=float)
                        ct_confmap_gt[-1, :, :] = 1.0
                        ct_obj_info = ct_obj_info_list[save_frame_idx % len(ct_files)]
                        if len(ct_obj_info) != 0:
                            ct_confmap_gt = generate_confmap(ct_obj_info)
                            ct_confmap_gt = normalize_confmap(ct_confmap_gt)
                            ct_confmap_gt = add_noise_channel(ct_confmap_gt)
                        assert ct_confmap_gt.shape == (
                            n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

                        # merge two RA slices: remove the static frequerncy in content and then add it to background
                        # translate the content data
                        if save_frame_idx % 30 == 0:
                            chose_chirp = randint(0, 127)
                            chose_chirp2 = chose_chirp + 127
                            # find a car in obj_info
                            need_info_car = find_obj_info('car', obj_info)
                            need_info_ped = find_obj_info('pedestrian', ct_obj_info)
                            need_info_cyc = find_obj_info('cyclist', ct_obj_info)

                            if need_info_car is None:
                                shift_angle = randint(-10, 10)
                                shift_range = randint(-10, 10)
                            else:
                                if need_info_ped is not None and need_info_cyc is not None:
                                    # random choose one
                                    if random() < 0.5:
                                        # choose ped
                                        need_info_ct = need_info_ped
                                    else:
                                        # choose cyclist
                                        need_info_ct = need_info_cyc
                                elif need_info_ped is not None:
                                    need_info_ct = need_info_ped
                                elif need_info_cyc is not None:
                                    need_info_ct = need_info_cyc
                                else:
                                    need_info_ct = None

                                if need_info_ct is None:
                                    shift_angle = randint(-10, 10)
                                    shift_range = randint(-10, 10)
                                else:
                                    diff_range = need_info_car[0] - need_info_ct[0]
                                    diff_angle = need_info_car[1] - need_info_ct[1]
                                    if abs(diff_range) < abs(diff_angle):
                                        # only shift_range
                                        shift_range = randint(diff_range - 2, diff_range + 2)
                                        shift_angle = 0
                                    else:
                                        shift_range = 0
                                        shift_angle = randint(diff_angle - 3, diff_angle + 3)

                            shift.append([shift_range, shift_angle])

                        ct_RA_slice, ct_RV_slice, ct_VA_slice, ct_confmap_gt = \
                            translation(ct_RA_slice, ct_RV_slice, ct_VA_slice, ct_confmap_gt, shift_range, shift_angle)
                        # print(type(RA_slice), type(RV_slice), type(VA_slice), type(confmap_gt))
                        # print(RA_slice.shape, RV_slice.shape, VA_slice.shape, confmap_gt.shape)
                        # input()

                        RA_slice = RA_slice + ct_RA_slice
                        RV_slice = RV_slice + ct_RV_slice
                        VA_slice = VA_slice + ct_VA_slice
                        confmap_gt = np.maximum(confmap_gt, ct_confmap_gt)
                        obj_info = obj_info + ct_obj_info

                        # save data
                        new_file_name = str(save_frame_idx).zfill(6) + '.npy'
                        new_imag_name = str(save_frame_idx).zfill(10) + '.jpg'
                        save_file_name_rv = save_dir_rv + '/' + new_file_name
                        save_file_name_va = save_dir_va + '/' + new_file_name
                        save_file_name_ra = save_dir_ra + '/' + new_file_name
                        save_file_name_ra2 = save_dir_ra2 + '/' + new_file_name
                        save_file_name_imag = save_dir_image + '/' + new_imag_name
                        # copy image and rename
                        shutil.copyfile(imag_dir, save_file_name_imag)
                        np.save(save_file_name_rv, RV_slice)
                        np.save(save_file_name_va, VA_slice)
                        np.save(save_file_name_ra, RA_slice[:, :, chose_chirp, :])
                        np.save(save_file_name_ra2, RA_slice[:, :, chose_chirp2, :])
                        # prepare files
                        path = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', "%04d", "%06d.npy")
                        detail_list[0].append(path)
                        confmap_list[0].append(confmap_gt)
                        confmap_list[1].append(obj_info)
                        print('finished ', file)
                        save_frame_idx += 1

            confmap_list[0] = np.array(confmap_list[0])
            # save pkl files
            pickle.dump(confmap_list, open(os.path.join(confmap_dir, set_type, save_seq + '.pkl'), 'wb'))
            # save pkl files
            pickle.dump(detail_list, open(os.path.join(detail_dir, set_type, save_seq + '.pkl'), 'wb'))
            save_seq_idx += 1

            # write to file
            if len(big_seq) > 1:
                print_seq = ' '
                for seq in big_seq:
                    print_seq = print_seq + seq + ' '
            else:
                print_seq = big_seq[0]
            with open(store_info_dir, 'w+') as filehandle:
                filehandle.write('content data: %s background data: %s\n' % (ct_seq, print_seq))
                for cid, listitem in enumerate(shift):
                    # frame_id, range_shift, angle_shift
                    filehandle.write('%d %d\n' % (listitem[0], listitem[1]))


def assemble_car_data():
    # 2019_04_09_cms1000: frame all
    # 2019_04_09_cms1001: frame all
    # 2019_04_09_cms1002: frame all
    # 2019_04_30_cm1s000: frame 210~300
    # 2019_04_30_pcms001: frame all
    # 2019_05_29_cm1s014: frame 40-180, 640-690
    # 2019_05_29_cm1s015: frame 80~430
    # 2019_05_29_cm1s016: frame 170~270
    # 2019_05_29_cm1s017: frame 335-500

    root_dir = '/media/admin-cmmb/Elements/CRdataset/'
    root_dir_store = '/mnt/sda/3DRadardata/'
    root_imag_dir = '/mnt/nas_crdataset/'
    data_dir = './data/'
    confmap_dir = os.path.join(data_dir, 'confmaps_gt')
    detail_dir = os.path.join(data_dir, 'data_details')
    set_type = 'train_new'
    seqs = [['2019_04_09_cms1000'], ['2019_04_09_cms1001'], ['2019_04_09_cms1002'], \
            ['2019_04_30_pcms001'], ['2019_04_30_cm1s000', '2019_05_29_cm1s014', \
            '2019_05_29_cm1s015', '2019_05_29_cm1s016', '2019_05_29_cm1s017']]
    useful_frames = [None, None, None, None, [[[210, 300]], [[40, 180], [640, 690]], \
                     [[80, 430]], [[170, 270]], [[335, 500]]]]
    save_capture_date = '2020_00_00'
    save_seq_idx = 0

    for idbs, big_seq in enumerate(seqs):
        biguseful_frame = useful_frames[idbs]
        save_frame_idx = 0
        ra_frame_offset = 0
        detail_list = [[], ra_frame_offset]
        confmap_list = [[], []]

        save_seq = save_capture_date + '_cm1s' + str(save_seq_idx).zfill(3)
        save_dir_rv = os.path.join(root_dir_store, save_capture_date, save_seq, 'RV_NPY')
        save_dir_ra = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', '0000')
        save_dir_ra2 = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', '0128')
        save_dir_va = os.path.join(root_dir_store, save_capture_date, save_seq, 'VA_NPY')
        save_dir_image = os.path.join(root_dir_store, save_capture_date, save_seq, 'images')
        if not os.path.exists(save_dir_rv):
            os.makedirs(save_dir_rv)
        if not os.path.exists(save_dir_ra):
            os.makedirs(save_dir_ra)
        if not os.path.exists(save_dir_ra2):
            os.makedirs(save_dir_ra2)
        if not os.path.exists(save_dir_va):
            os.makedirs(save_dir_va)
        if not os.path.exists(save_dir_image):
            os.makedirs(save_dir_image)

        print('Making sequence: ', save_seq)
        for idx, seq in enumerate(big_seq):
            capture_date = seq[0:10]
            seq_dir = os.path.join(root_dir, capture_date, seq, 'WIN_R_MAT')
            seq_path = os.path.join(root_imag_dir, capture_date, seq)
            seq_image_dir = os.path.join(root_imag_dir, capture_date, seq, 'images')

            files = sorted(os.listdir(seq_dir))
            images = sorted(os.listdir(seq_image_dir))
            print('Processing ', seq)
            if biguseful_frame is not None:
                useful_frame = biguseful_frame[idx]
                for segm in useful_frame:
                    start_frame = segm[0]
                    end_frame = segm[1]
                    print('Assemble frames:', start_frame, ' ~ ', end_frame, 'in this squence')
                    # process, save data and prepare data
                    for idf, file in enumerate(files[start_frame:end_frame]):
                        file_dir = os.path.join(seq_dir, file)
                        imag_dir = os.path.join(seq_image_dir, images[idf+start_frame])
                        mat = spio.loadmat(file_dir, squeeze_me=True)
                        data = np.asarray(mat["R_data"])
                        RV_slice, rv_raw1, rv_raw2 = produce_RV_slice(data)
                        VA_slice = produce_VA_slice(rv_raw1, rv_raw2)
                        RA_slice = produce_RA_slice(data)
                        # save data
                        new_file_name = str(save_frame_idx).zfill(6) + '.npy'
                        new_imag_name = str(save_frame_idx).zfill(10) + '.jpg'
                        save_file_name_rv = save_dir_rv + '/' + new_file_name
                        save_file_name_va = save_dir_va + '/' + new_file_name
                        save_file_name_ra = save_dir_ra + '/' + new_file_name
                        save_file_name_ra2 = save_dir_ra2 + '/' + new_file_name
                        save_file_name_imag = save_dir_image + '/' + new_imag_name
                        # copy image and rename
                        shutil.copyfile(imag_dir, save_file_name_imag)
                        np.save(save_file_name_rv, RV_slice)
                        np.save(save_file_name_va, VA_slice)
                        np.save(save_file_name_ra, RA_slice[:,:,0,:])
                        np.save(save_file_name_ra2, RA_slice[:, :, 128, :])
                        # prepare files
                        path = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', "%04d", "%06d.npy")
                        detail_list[0].append(path)
                        print('finished ', file)
                        save_frame_idx += 1

                    # use labelled RAMap and prepare ground truth
                    try:
                        obj_info_list = read_ra_labels_csv(seq_path)
                    except Exception as e:
                        print("Load sequence %s failed!" % seq_path)
                        print(e)
                        continue

                    for obj_info in obj_info_list[start_frame:end_frame]:
                        confmap_gt = np.zeros((n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                                              dtype=float)
                        confmap_gt[-1, :, :] = 1.0
                        if len(obj_info) != 0:
                            confmap_gt = generate_confmap(obj_info)
                            confmap_gt = normalize_confmap(confmap_gt)
                            confmap_gt = add_noise_channel(confmap_gt)
                        assert confmap_gt.shape == (n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

                        confmap_list[0].append(confmap_gt)
                        confmap_list[1].append(obj_info)
                        # end objects loop
                    assert len(confmap_list[0]) == len(detail_list[0])

            else:
                print('Assemble all frames in this squence')
                for idf, file in enumerate(files):
                    file_dir = os.path.join(seq_dir, file)
                    imag_dir = os.path.join(seq_image_dir, images[idf])
                    mat = spio.loadmat(file_dir, squeeze_me=True)
                    data = np.asarray(mat["R_data"])
                    RV_slice, rv_raw1, rv_raw2 = produce_RV_slice(data)
                    VA_slice = produce_VA_slice(rv_raw1, rv_raw2)
                    RA_slice = produce_RA_slice(data)
                    # save data
                    new_file_name = str(save_frame_idx).zfill(6) + '.npy'
                    new_imag_name = str(save_frame_idx).zfill(10) + '.jpg'
                    save_file_name_rv = save_dir_rv + '/' + new_file_name
                    save_file_name_va = save_dir_va + '/' + new_file_name
                    save_file_name_ra = save_dir_ra + '/' + new_file_name
                    save_file_name_ra2 = save_dir_ra2 + '/' + new_file_name
                    save_file_name_imag = save_dir_image + '/' + new_imag_name
                    # copy image and rename
                    shutil.copyfile(imag_dir, save_file_name_imag)
                    np.save(save_file_name_rv, RV_slice)
                    np.save(save_file_name_va, VA_slice)
                    np.save(save_file_name_ra, RA_slice[:, :, 0, :])
                    np.save(save_file_name_ra2, RA_slice[:, :, 128, :])
                    # prepare files
                    path = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', "%04d", "%06d.npy")
                    detail_list[0].append(path)
                    print('finished ', file)
                    save_frame_idx += 1

                # use labelled RAMap and prepare ground truth
                try:
                    obj_info_list = read_ra_labels_csv(seq_path)
                except Exception as e:
                    print("Load sequence %s failed!" % seq_path)
                    print(e)
                    continue

                for obj_info in obj_info_list:
                    confmap_gt = np.zeros((n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                                          dtype=float)
                    confmap_gt[-1, :, :] = 1.0
                    if len(obj_info) != 0:
                        confmap_gt = generate_confmap(obj_info)
                        confmap_gt = normalize_confmap(confmap_gt)
                        confmap_gt = add_noise_channel(confmap_gt)
                    assert confmap_gt.shape == (n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

                    confmap_list[0].append(confmap_gt)
                    confmap_list[1].append(obj_info)
                    # end objects loop
                assert len(confmap_list[0]) == len(detail_list[0])

        confmap_list[0] = np.array(confmap_list[0])
        # save pkl files
        pickle.dump(confmap_list, open(os.path.join(confmap_dir, set_type, save_seq + '.pkl'), 'wb'))
        # save pkl files
        pickle.dump(detail_list, open(os.path.join(detail_dir, set_type, save_seq + '.pkl'), 'wb'))
        save_seq_idx += 1


def generate_car_nearborder_data():
    # 2019_04_09_cms1000: frame all
    # 2019_04_09_cms1001: frame all
    # 2019_04_09_cms1002: frame all
    # 2019_04_30_cm1s000: frame 210~300
    # 2019_04_30_pcms001: frame all
    # 2019_05_29_cm1s014: frame 40-180, 640-690
    # 2019_05_29_cm1s015: frame 80~430
    # 2019_05_29_cm1s016: frame 170~270
    # 2019_05_29_cm1s017: frame 335-500

    root_dir = '/media/admin-cmmb/Elements/CRdataset/'
    root_dir_store = '/mnt/sda/3DRadardata/'
    root_imag_dir = '/mnt/nas_crdataset/'
    data_dir = './data/'
    confmap_dir = os.path.join(data_dir, 'confmaps_gt')
    detail_dir = os.path.join(data_dir, 'data_details')
    set_type = 'train_new'
    # seqs = [['2019_04_09_cms1000'], ['2019_04_09_cms1001'], ['2019_04_09_cms1002'], \
    #         ['2019_04_30_pcms001'], ['2019_04_30_cm1s000', '2019_05_29_cm1s014', \
    #         '2019_05_29_cm1s015', '2019_05_29_cm1s016', '2019_05_29_cm1s017']]
    # useful_frames = [None, None, None, None, [[[210, 300]], [[40, 180], [640, 690]], \
    #                  [[80, 430]], [[170, 270]], [[335, 500]]]]
    seqs = [['2019_04_09_cms1000'], ['2019_04_09_cms1001'], ['2019_04_09_cms1002']]
    useful_frames = [None, None, None]
    save_capture_date = '2020_00_00'
    save_seq_idx = 5

    for idbs, big_seq in enumerate(seqs):
        biguseful_frame = useful_frames[idbs]
        save_frame_idx = 0
        ra_frame_offset = 0
        detail_list = [[], ra_frame_offset]
        confmap_list = [[], []]

        save_seq = save_capture_date + '_cm1s' + str(save_seq_idx).zfill(3)
        save_dir_rv = os.path.join(root_dir_store, save_capture_date, save_seq, 'RV_NPY')
        save_dir_ra = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', '0000')
        save_dir_ra2 = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', '0128')
        save_dir_va = os.path.join(root_dir_store, save_capture_date, save_seq, 'VA_NPY')
        save_dir_image = os.path.join(root_dir_store, save_capture_date, save_seq, 'images')
        if not os.path.exists(save_dir_rv):
            os.makedirs(save_dir_rv)
        if not os.path.exists(save_dir_ra):
            os.makedirs(save_dir_ra)
        if not os.path.exists(save_dir_ra2):
            os.makedirs(save_dir_ra2)
        if not os.path.exists(save_dir_va):
            os.makedirs(save_dir_va)
        if not os.path.exists(save_dir_image):
            os.makedirs(save_dir_image)

        print('Making sequence: ', save_seq)
        for idx, seq in enumerate(big_seq):
            capture_date = seq[0:10]
            seq_dir = os.path.join(root_dir, capture_date, seq, 'WIN_R_MAT')
            seq_path = os.path.join(root_imag_dir, capture_date, seq)
            seq_image_dir = os.path.join(root_imag_dir, capture_date, seq, 'images')

            files = sorted(os.listdir(seq_dir))
            images = sorted(os.listdir(seq_image_dir))
            print('Processing ', seq)

            # use labelled RAMap and prepare ground truth
            try:
                obj_info_list = read_ra_labels_csv(seq_path)
            except Exception as e:
                print("Load sequence %s failed!" % seq_path)
                print(e)
                continue

            if len(obj_info_list) <= len(files):
                files = files[len(files) - len(obj_info_list):]
            else:
                raise ValueError

            if biguseful_frame is not None:
                useful_frame = biguseful_frame[idx]
                for segm in useful_frame:
                    start_frame = segm[0]
                    end_frame = segm[1]
                    print('Assemble frames:', start_frame, ' ~ ', end_frame, 'in this squence')
                    # process, save data and prepare data
                    for idf, file in enumerate(files[start_frame:end_frame]):
                        file_dir = os.path.join(seq_dir, file)
                        imag_dir = os.path.join(seq_image_dir, images[idf+start_frame])
                        mat = spio.loadmat(file_dir, squeeze_me=True)
                        data = np.asarray(mat["R_data"])
                        RV_slice, rv_raw1, rv_raw2 = produce_RV_slice(data)
                        VA_slice = produce_VA_slice(rv_raw1, rv_raw2)
                        RA_slice = produce_RA_slice(data)
                        # generate confidence map
                        obj_info = obj_info_list[idf + start_frame]
                        confmap_gt = np.zeros((n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                                              dtype=float)
                        confmap_gt[-1, :, :] = 1.0
                        if len(obj_info) != 0:
                            confmap_gt = generate_confmap(obj_info)
                            confmap_gt = normalize_confmap(confmap_gt)
                            confmap_gt = add_noise_channel(confmap_gt)
                        assert confmap_gt.shape == (
                        n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

                        if save_frame_idx % 30 == 0: # TODO: update save_frame_idx
                            chose_chirp = randint(0, 127)
                            chose_chirp2 = chose_chirp + 127
                            # choose one car and decide the shift_range and shift_angle
                            need_info = find_obj_info('car', obj_info)
                            if need_info is not None:
                                obj_rng = need_info[0]
                                obj_agl = need_info[1]
                                # angle_shift
                                if obj_agl < 128 - obj_rng:
                                    shift_range = 0
                                    shift_angle = randint(-obj_agl-4, -obj_agl+4)
                                # range_shift
                                else:
                                    shift_range = randint(128-obj_rng-15, 128-obj_rng)
                                    shift_angle = 0
                            else:
                                shift_range = 0
                                shift_angle = 0

                        if shift_range != 0 and shift_angle != 0:
                            RA_slice, RV_slice, VA_slice, confmap_gt = \
                                translation(RA_slice, RV_slice, VA_slice, confmap_gt, shift_range, shift_angle)

                        # save data
                        new_file_name = str(save_frame_idx).zfill(6) + '.npy'
                        new_imag_name = str(save_frame_idx).zfill(10) + '.jpg'
                        save_file_name_rv = save_dir_rv + '/' + new_file_name
                        save_file_name_va = save_dir_va + '/' + new_file_name
                        save_file_name_ra = save_dir_ra + '/' + new_file_name
                        save_file_name_ra2 = save_dir_ra2 + '/' + new_file_name
                        save_file_name_imag = save_dir_image + '/' + new_imag_name
                        # copy image and rename
                        shutil.copyfile(imag_dir, save_file_name_imag)
                        np.save(save_file_name_rv, RV_slice)
                        np.save(save_file_name_va, VA_slice)
                        np.save(save_file_name_ra, RA_slice[:, :, chose_chirp, :])
                        np.save(save_file_name_ra2, RA_slice[:, :, chose_chirp2, :])
                        # prepare files
                        path = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', "%04d", "%06d.npy")
                        detail_list[0].append(path)
                        confmap_list[0].append(confmap_gt)
                        confmap_list[1].append(obj_info)
                        # end objects loop
                        print('finished ', file)
                        save_frame_idx += 1
            else:
                print('Assemble all frames in this squence')
                for idf, file in enumerate(files):
                    file_dir = os.path.join(seq_dir, file)
                    imag_dir = os.path.join(seq_image_dir, images[idf])
                    mat = spio.loadmat(file_dir, squeeze_me=True)
                    data = np.asarray(mat["R_data"])
                    RV_slice, rv_raw1, rv_raw2 = produce_RV_slice(data)
                    VA_slice = produce_VA_slice(rv_raw1, rv_raw2)
                    RA_slice = produce_RA_slice(data)
                    # generate confidence map
                    obj_info = obj_info_list[idf]
                    confmap_gt = np.zeros((n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                                          dtype=float)
                    confmap_gt[-1, :, :] = 1.0
                    if len(obj_info) != 0:
                        confmap_gt = generate_confmap(obj_info)
                        confmap_gt = normalize_confmap(confmap_gt)
                        confmap_gt = add_noise_channel(confmap_gt)
                    assert confmap_gt.shape == (
                        n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

                    if save_frame_idx % 30 == 0:  # TODO: update save_frame_idx
                        chose_chirp = randint(0, 127)
                        chose_chirp2 = chose_chirp + 127
                        # choose one car and decide the shift_range and shift_angle
                        need_info = find_obj_info('car', obj_info)
                        if need_info is not None:
                            obj_rng = need_info[0]
                            obj_agl = need_info[1]
                            # angle_shift
                            if obj_agl < 128 - obj_rng:
                                shift_range = 0
                                shift_angle = randint(-obj_agl - 4, -obj_agl + 4)
                            # range_shift
                            else:
                                shift_range = randint(128 - obj_rng - 15, 128 - obj_rng)
                                shift_angle = 0
                        else:
                            shift_range = 0
                            shift_angle = 0

                    if shift_range != 0 and shift_angle != 0:
                        RA_slice, RV_slice, VA_slice, confmap_gt = \
                            translation(RA_slice, RV_slice, VA_slice, confmap_gt, shift_range, shift_angle)

                    # save data
                    new_file_name = str(save_frame_idx).zfill(6) + '.npy'
                    new_imag_name = str(save_frame_idx).zfill(10) + '.jpg'
                    save_file_name_rv = save_dir_rv + '/' + new_file_name
                    save_file_name_va = save_dir_va + '/' + new_file_name
                    save_file_name_ra = save_dir_ra + '/' + new_file_name
                    save_file_name_ra2 = save_dir_ra2 + '/' + new_file_name
                    save_file_name_imag = save_dir_image + '/' + new_imag_name
                    # copy image and rename
                    shutil.copyfile(imag_dir, save_file_name_imag)
                    np.save(save_file_name_rv, RV_slice)
                    np.save(save_file_name_va, VA_slice)
                    np.save(save_file_name_ra, RA_slice[:, :, chose_chirp, :])
                    np.save(save_file_name_ra2, RA_slice[:, :, chose_chirp2, :])
                    # prepare files
                    path = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', "%04d", "%06d.npy")
                    detail_list[0].append(path)
                    confmap_list[0].append(confmap_gt)
                    confmap_list[1].append(obj_info)
                    # end objects loop
                    print('finished ', file)
                    save_frame_idx += 1
                    # end objects loop
        assert len(confmap_list[0]) == len(detail_list[0])
        confmap_list[0] = np.array(confmap_list[0])
        # save pkl files
        pickle.dump(confmap_list, open(os.path.join(confmap_dir, set_type, save_seq + '.pkl'), 'wb'))
        # save pkl files
        pickle.dump(detail_list, open(os.path.join(detail_dir, set_type, save_seq + '.pkl'), 'wb'))
        save_seq_idx += 1


def mix_multiple_data():
    # background
    # 2019_05_29_mlms006: frame all
    # 2019_05_29_pcms005: frame all
    # 2019_05_29_bcms000: frame all
    # 2019_05_29_cm1s014: frame all
    # 2019_05_29_cm1s015: frame all
    # 2019_05_29_cm1s016: frame all
    # 2019_05_29_cm1s017: frame all

    # content
    # 2019_05_29_pm3s000: frame all
    # 2019_05_29_pbms007: frame all
    # 2019_05_29_bm1s018: frame all
    # 2019_04_30_pm2s003: frame all
    # 2019_04_30_mlms000: frame all
    # 2019_04_30_mlms001: frame all
    # 2019_04_30_mlms002: frame all
    # 2019_04_09_pms2000: frame all

    root_dir = '/media/admin-cmmb/Elements/CRdataset/'
    root_dir_store = '/mnt/sda/3DRadardata/'
    root_imag_dir = '/mnt/nas_crdataset/'
    data_dir = './data/'
    confmap_dir = os.path.join(data_dir, 'confmaps_gt')
    detail_dir = os.path.join(data_dir, 'data_details')
    set_type = 'train_new'
    bg_seqs = ['2019_05_29_mlms006', '2019_05_29_pcms005', '2019_05_29_bcms000', \
               '2019_05_29_cm1s014', '2019_05_29_cm1s015', '2019_05_29_cm1s016', \
               '2019_05_29_cm1s017']
    # bg_seqs.reverse()
    ct_seqs = ['2019_05_29_pm3s000', '2019_05_29_pbms007', '2019_05_29_bm1s018', \
               '2019_04_30_pm2s003', '2019_04_30_mlms000', '2019_04_30_mlms001', \
               '2019_04_30_mlms002', '2019_04_09_pms2000']
    position_shift = [None, None, ['left'], None, None, None, None, None]
    save_seq_idx = 0 # TODO: change it accroding to scenario

    for idx, seq in enumerate(ct_seqs):
        bg_seq = bg_seqs[idx % len(bg_seqs)]
        print('content data', seq, 'background data', bg_seq)
        capture_date = seq[0:10]
        bg_capture_date = bg_seq[0:10]
        save_capture_date = '2020_00_00'
        save_seq = '2020_00_00_mlms' + str(save_seq_idx).zfill(3) # TODO: update save_seq_index
        seq_dir = os.path.join(root_dir, capture_date, seq, 'WIN_R_MAT')
        bg_seq_dir = os.path.join(root_dir, bg_capture_date, bg_seq, 'WIN_R_MAT')
        seq_path = os.path.join(root_imag_dir, capture_date, seq)
        bg_seq_path = os.path.join(root_imag_dir, bg_capture_date, bg_seq)
        seq_image_dir = os.path.join(root_imag_dir, capture_date, seq, 'images')

        ra_frame_offset = 0
        detail_list = [[], ra_frame_offset]
        confmap_list = [[], []]
        shift = []

        save_dir_rv = os.path.join(root_dir_store, save_capture_date, save_seq, 'RV_NPY')
        save_dir_ra = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', '0000')
        save_dir_ra2 = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', '0128')
        save_dir_va = os.path.join(root_dir_store, save_capture_date, save_seq, 'VA_NPY')
        save_dir_image = os.path.join(root_dir_store, save_capture_date, save_seq, 'images')
        store_info_dir = os.path.join(root_dir_store, save_capture_date, save_seq) + '/seq_info.txt'
        if not os.path.exists(save_dir_rv):
            os.makedirs(save_dir_rv)
        if not os.path.exists(save_dir_ra):
            os.makedirs(save_dir_ra)
        if not os.path.exists(save_dir_ra2):
            os.makedirs(save_dir_ra2)
        if not os.path.exists(save_dir_va):
            os.makedirs(save_dir_va)
        if not os.path.exists(save_dir_image):
            os.makedirs(save_dir_image)
        files = sorted(os.listdir(seq_dir))
        bg_files = sorted(os.listdir(bg_seq_dir))
        images = sorted(os.listdir(seq_image_dir))
        print('Processing ', seq)
        valid_len = min(len(bg_files), len(files))

        # read label
        try:
            obj_info_list = read_ra_labels_csv(seq_path)
        except Exception as e:
            print("Load sequence %s failed!" % seq_path)
            continue
        try:
            bg_obj_info_list = read_ra_labels_csv(bg_seq_path)
        except Exception as e:
            print("Load sequence %s failed!" % bg_seq_path)
            continue

        for idf, file in enumerate(files[0:valid_len]):
            file_dir = os.path.join(seq_dir, file)
            bg_file_dir = os.path.join(bg_seq_dir, bg_files[idf])
            imag_dir = os.path.join(seq_image_dir, images[idf])

            mat = spio.loadmat(file_dir, squeeze_me=True)
            data = np.asarray(mat["R_data"])
            data = data - np.mean(data, axis=2, keepdims=True) # remove static components
            RA_slice = produce_RA_slice(data)
            RV_slice, rv_raw1, rv_raw2 = produce_RV_slice(data)
            VA_slice = produce_VA_slice(rv_raw1, rv_raw2)
            # generate confidence map
            obj_info = obj_info_list[idf]
            confmap_gt = np.zeros((n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                                  dtype=float)
            confmap_gt[-1, :, :] = 1.0
            if len(obj_info) != 0:
                confmap_gt = generate_confmap(obj_info)
                confmap_gt = normalize_confmap(confmap_gt)
                confmap_gt = add_noise_channel(confmap_gt)
            assert confmap_gt.shape == (n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

            bg_mat = spio.loadmat(bg_file_dir, squeeze_me=True)
            bg_data = np.asarray(bg_mat["R_data"])
            bg_RA_slice = produce_RA_slice(bg_data)
            bg_RV_slice, rv_raw1, rv_raw2 = produce_RV_slice(bg_data)
            bg_VA_slice = produce_VA_slice(rv_raw1, rv_raw2)
            # generate confidence map
            bg_obj_info = bg_obj_info_list[idf]
            bg_confmap_gt = np.zeros((n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                                  dtype=float)
            bg_confmap_gt[-1, :, :] = 1.0
            if len(bg_obj_info) != 0:
                bg_confmap_gt = generate_confmap(bg_obj_info)
                bg_confmap_gt = normalize_confmap(bg_confmap_gt)
                bg_confmap_gt = add_noise_channel(bg_confmap_gt)
            assert bg_confmap_gt.shape == (n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

            # merge two RA slices: remove the static frequerncy in content and then add it to background
            # translate the content data
            if idf % 100 == 0:
                if position_shift[idx] is None:
                    # random translate left and upword
                    shift_angle = randint(-20, 5)
                    shift_range = randint(-5, 30)
                elif position_shift[idx][0] == 'left':
                    shift_angle = randint(-20, 5)
                    shift_range = randint(-5, 5)
                else:
                    raise ValueError
                chose_chirp = randint(0, 127)
                chose_chirp2 = chose_chirp + 127
                shift.append([shift_range, shift_angle])

            RA_slice, RV_slice, VA_slice, confmap_gt = translation(RA_slice, RV_slice, VA_slice, confmap_gt, shift_range, shift_angle)
            # print(type(RA_slice), type(RV_slice), type(VA_slice), type(confmap_gt))
            # print(RA_slice.shape, RV_slice.shape, VA_slice.shape, confmap_gt.shape)
            # input()

            RA_slice = RA_slice + bg_RA_slice
            RV_slice = RV_slice + bg_RV_slice
            VA_slice = VA_slice + bg_VA_slice
            confmap_gt = np.maximum(confmap_gt, bg_confmap_gt)
            obj_info = obj_info + bg_obj_info

            # save data
            new_file_name = str(idf).zfill(6) + '.npy'
            new_imag_name = str(idf).zfill(10) + '.jpg'
            save_file_name_rv = save_dir_rv + '/' + new_file_name
            save_file_name_va = save_dir_va + '/' + new_file_name
            save_file_name_ra = save_dir_ra + '/' + new_file_name
            save_file_name_ra2 = save_dir_ra2 + '/' + new_file_name
            save_file_name_imag = save_dir_image + '/' + new_imag_name
            # copy image and rename
            shutil.copyfile(imag_dir, save_file_name_imag)
            np.save(save_file_name_rv, RV_slice)
            np.save(save_file_name_va, VA_slice)
            np.save(save_file_name_ra, RA_slice[:, :, chose_chirp, :])
            np.save(save_file_name_ra2, RA_slice[:, :, chose_chirp2, :])
            # prepare files
            path = os.path.join(root_dir_store, save_capture_date, save_seq, 'RA_NPY', "%04d", "%06d.npy")
            detail_list[0].append(path)
            confmap_list[0].append(confmap_gt)
            confmap_list[1].append(obj_info)
            print('finished ', file)

        save_seq_idx += 1
        confmap_list[0] = np.array(confmap_list[0])
        # save pkl files
        pickle.dump(confmap_list, open(os.path.join(confmap_dir, set_type, save_seq + '.pkl'), 'wb'))
        # save pkl files
        pickle.dump(detail_list, open(os.path.join(detail_dir, set_type, save_seq + '.pkl'), 'wb'))
        # save data information and shift information
        # write to file
        with open(store_info_dir, 'w+') as filehandle:
            filehandle.write('content data: %s background data: %s\n' % (seq, bg_seq))
            for cid, listitem in enumerate(shift):
                # frame_id, range_shift, angle_shift
                filehandle.write('%d %d\n' % (listitem[0], listitem[1]))



if __name__ == '__main__':
    # test
    # assemble_car_data()
   # mix_multiple_data()
   # generate_mix_car_pedcyc()
   # generate_mix_ped()
   # generate_car_nearborder_data()
   # generate_mix_staticped()
   # generate_mixcross_car_pedcyc()
   # generate_mix_ped_cyc()
   generate_mix_staticcar_pedcyc()