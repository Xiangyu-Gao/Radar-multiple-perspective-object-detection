import os

from utils.dataset_tools import calculate_frame_offset


if __name__ == '__main__':
    data_root = '/mnt/nas_crdataset'
    dates = ['2019_09_18']
    for date in dates:
        seqs = sorted(os.listdir(os.path.join(data_root, date)))
        for seq in seqs:
            seq_path = os.path.join(data_root, date, seq)
            start_time_txt = os.path.join(seq_path, 'start_time.txt')
            offset01, offset02, offset12 = calculate_frame_offset(start_time_txt)
            print("%s: %d, %d, %d" % (seq, offset01, offset02, offset12))
