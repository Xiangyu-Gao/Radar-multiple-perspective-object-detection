import numpy as np
import scipy.io as spio
import os
import math
from utils.read_annotations import read_ra_labels_csv, read_3d_labels_refine_txt
from utils.dataset_tools import calculate_frame_offset
from dataPrepare.relocate_dataset import produce_RA_slice

n_angle = 128
n_chirp = 255
n_vel = 256
n_rx = 8
Is_rm_static = True

def velocity_fft(data):
    hanning_win = np.hamming(n_chirp)
    win_data1 = np.zeros([data.shape[0], data.shape[1], data.shape[2]], dtype=np.complex128)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            win_data1[i, j, :] = np.multiply(data[i, j, :], hanning_win)

    fft_data_raw1 = np.fft.fft(win_data1, n_vel, axis=2)
    fft_data_raw1 = np.fft.fftshift(fft_data_raw1, axes=2)

    return fft_data_raw1


def main():
    root_dir = '/media/admin-cmmb/Elements/CRdataset/'
    root_dir_store = '/mnt/sda/Labels/'
    save_file_name = '/v_label.txt'
    # dates = ['2019_04_09', '2019_04_30', '2019_05_09', '2019_05_23', '2019_05_28', '2019_05_29']
    dates = ['2019_09_29']
    useful_seqs = sorted(os.listdir('./data/data_details/train_all/')) + sorted(os.listdir('./data/data_details/test/'))
    for capture_date in dates:
        cap_folder_dir = os.path.join(root_dir, capture_date)
        seqs = sorted(os.listdir(cap_folder_dir))
        for seq in seqs:
            if seq + '.pkl' in useful_seqs:
                Is_first_open = True
                seq_dir = os.path.join(cap_folder_dir, seq, 'WIN_R_MAT')
                save_dir = os.path.join(root_dir_store, capture_date, seq)
                save_file_dir = save_dir + save_file_name
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                files = sorted(os.listdir(seq_dir))
                print('Processing ', seq)

                seq_mon = int(seq.split('_')[1])
                label_path = os.path.join('/mnt/nas_crdataset', capture_date, seq)
                start_id = int(files[0].split('.mat')[0].split('_')[-1])

                if seq_mon < 9:
                    # read detection results from human label
                    labels = read_ra_labels_csv(label_path)
                    assert len(files) == len(labels)
                    for idf, file in enumerate(files):
                        file_dir = os.path.join(seq_dir, file)
                        mat = spio.loadmat(file_dir, squeeze_me=True)
                        data = np.asarray(mat["R_data"])
                        ra_cmplx = produce_RA_slice(data, keep_complex=True)
                        if Is_rm_static is True:
                            ra_cmplx = ra_cmplx - np.sum(ra_cmplx, axis=2, keepdims=True) / ra_cmplx.shape[2]
                            label_frame = labels[idf]
                            for label in label_frame:
                                rng_idx = label[0]
                                agl_idx = label[1]
                                class_id = label[2]
                                if class_id >= 0:
                                    chirp_data = ra_cmplx[rng_idx-1:rng_idx+2, agl_idx-1:agl_idx+2, :]
                                    rav_data = np.abs(velocity_fft(chirp_data)) # 256 points velocity fft
                                    _, _, dop_idx = np.unravel_index(rav_data.argmax(), rav_data.shape) # maximum dop idx is 256
                                    dop_idx = math.floor(dop_idx / 2)
                                    if Is_first_open:
                                        with open(save_file_dir, 'w+') as filehandle:
                                            # frame_id, range_id, angle_id, class_id, doppler_id
                                            filehandle.write('%d %d %d %d %d\n' % (start_id+idf,
                                                                                   rng_idx, agl_idx, class_id, dop_idx))
                                        Is_first_open = False
                                    else:
                                        with open(save_file_dir, 'a+') as filehandle:
                                            # frame_id, range_id, angle_id, class_id, doppler_id
                                            filehandle.write('%d %d %d %d %d\n' % (start_id+idf,
                                                                                   rng_idx, agl_idx, class_id, dop_idx))

                ra_frame_offset = calculate_frame_offset(os.path.join(label_path, 'start_time.txt'))[0]
                if ra_frame_offset > 0 and seq_mon >= 9:
                    all_labels = sorted(os.listdir(os.path.join(label_path, 'dets_refine')))
                    ra_frame_offset += 40
                    start_frame_id = ra_frame_offset  # radar start id,  # camera start id = 0
                    assert start_frame_id == len(files) - len(all_labels)
                if seq_mon >= 9:
                    for idf, file in enumerate(files):
                        if idf < start_frame_id:
                            continue
                        file_dir = os.path.join(seq_dir, file)
                        mat = spio.loadmat(file_dir, squeeze_me=True)
                        data = np.asarray(mat["R_data"])
                        ra_cmplx = produce_RA_slice(data, keep_complex=True)
                        if Is_rm_static is True:
                            ra_cmplx = ra_cmplx - np.sum(ra_cmplx, axis=2, keepdims=True) / ra_cmplx.shape[2]
                            label_frame_name = all_labels[idf-start_frame_id]
                            label_frame = read_3d_labels_refine_txt(label_path, label_frame_name)
                            for label in label_frame:
                                rng_idx = label[0]
                                agl_idx = label[1]
                                class_id = label[2]
                                if class_id >= 0:
                                    chirp_data = ra_cmplx[rng_idx - 1:rng_idx + 2, agl_idx - 1:agl_idx + 2, :]
                                    rav_data = np.abs(velocity_fft(chirp_data))  # 256 points velocity fft
                                    _, _, dop_idx = np.unravel_index(rav_data.argmax(), rav_data.shape)  # maximum dop idx is 256
                                    dop_idx = math.floor(dop_idx / 2)
                                    if Is_first_open:
                                        with open(save_file_dir, 'w+') as filehandle:
                                            # frame_id, range_id, angle_id, class_id, doppler_id
                                            filehandle.write('%d %d %d %d %d\n' % (idf, rng_idx, agl_idx, class_id, dop_idx))
                                        Is_first_open = False
                                    else:
                                        with open(save_file_dir, 'a+') as filehandle:
                                            # frame_id, range_id, angle_id, class_id, doppler_id
                                            filehandle.write('%d %d %d %d %d\n' % (idf, rng_idx, agl_idx, class_id, dop_idx))


if __name__ == '__main__':
    main()


