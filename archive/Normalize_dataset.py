import numpy as np
import os
import math
import matplotlib.pyplot as plt
from random import random
from utils import find_nearest

interval = 0.0001
# thres = 0.0078
thres = 0.05
# norm_pers = 'RV'
norm_pers = 'VA'

def summarize():
    save_hist_dir = 'hist'
    save_edge_dir = 'bin_edge'
    # dates = ['2019_04_09', '2019_04_30', '2019_05_09', '2019_05_28', '2019_05_29']
    dates = ['2019_09_08', '2019_09_18', '2019_09_29']
    bin_min = 0
    bin_max = 0
    for capture_date in dates:
        if norm_pers == 'RA':
            save_edge_dir_seq = './data/data_hist/' + save_edge_dir + '_' + capture_date + '.npy'
            save_hist_dir_seq = './data/data_hist/' + save_hist_dir + '_' + capture_date + '.npy'
        else:
            save_edge_dir_seq = './data/data_hist/' + save_edge_dir + '_' + capture_date + '_' + norm_pers + '.npy'
            save_hist_dir_seq = './data/data_hist/' + save_hist_dir + '_' + capture_date + '_' + norm_pers + '.npy'
        # print(save_hist_dir_seq)
        hist = np.load(save_hist_dir_seq)
        bin_edges = np.load(save_edge_dir_seq)
        bin_min = min(bin_min, bin_edges[0])
        bin_max = max(bin_max, bin_edges[-1])
    # print(bin_min)
    # print(bin_max)

    # create new whole bin edges and histogram
    whole_bin_edges = np.arange(round(bin_min / interval), round(bin_max / interval) + 1, 1) * interval
    whole_hist = np.zeros(whole_bin_edges.shape[0] - 1)
    for capture_date in dates:
        if norm_pers == 'RA':
            save_edge_dir_seq = './data/data_hist/' + save_edge_dir + '_' + capture_date + '.npy'
            save_hist_dir_seq = './data/data_hist/' + save_hist_dir + '_' + capture_date + '.npy'
        else:
            save_edge_dir_seq = './data/data_hist/' + save_edge_dir + '_' + capture_date + '_' + norm_pers + '.npy'
            save_hist_dir_seq = './data/data_hist/' + save_hist_dir + '_' + capture_date + '_' + norm_pers + '.npy'
        hist = np.load(save_hist_dir_seq)
        bin_edges = np.load(save_edge_dir_seq)
        idx_min, _ = find_nearest(whole_bin_edges[0:-1], bin_edges[0])
        whole_hist[idx_min:idx_min + hist.shape[0]] += hist

    # plot the distribution
    hist_cdf = np.cumsum(whole_hist) / np.sum(whole_hist)
    idx_05, _ = find_nearest(hist_cdf, 0.0015)
    idx_95, _ = find_nearest(hist_cdf, 0.9985)
    print('#################### ', norm_pers, ' #####################')
    print('Capture data date', dates)
    print(idx_05, whole_bin_edges[idx_05])
    print(idx_95, whole_bin_edges[idx_95])
    print(hist_cdf)
    print(bin_min, bin_max)
    # print(hist_cdf.shape)
    # print(whole_hist.shape)
    #
    # plt.plot(whole_bin_edges[0:-1], whole_hist/sum(whole_hist))
    # # plt.xlim(-5, 5)
    # plt.show()
    #
    # # calculate mean and variance
    mean = np.sum(np.multiply(whole_bin_edges[0:-1], whole_hist / sum(whole_hist)))
    variance = np.sum(np.multiply(np.square(whole_bin_edges[0:-1] - mean), whole_hist)) / (sum(whole_hist) - 1)
    print(mean)
    print(math.sqrt(variance))
    #
    # plt.plot((whole_bin_edges[0:-1]-mean)/math.sqrt(variance), whole_hist/sum(whole_hist))
    # plt.xlim(-5, 5)
    # plt.show()


def generate_hist(bin_edges, hist, whole_bin_edges, whole_hist, int_initmin, int_initmax):

    if bin_edges[0] >= int_initmin and bin_edges[-1] <= int_initmax:
        idx_min, _ = find_nearest(whole_bin_edges[0:-1], bin_edges[0])
        whole_hist[idx_min:idx_min + hist.shape[0]] += hist

    elif bin_edges[0] < int_initmin and bin_edges[-1] <= int_initmax:
        # creaet the new store list and copy the old one to it.
        added_bin_edges_left = np.arange(round(bin_edges[0] / interval),
                                         round(int_initmin / interval), 1) * interval
        added_hist_left = np.zeros_like(added_bin_edges_left)
        whole_bin_edges = np.concatenate((added_bin_edges_left, whole_bin_edges), axis=0)
        whole_hist = np.concatenate((added_hist_left, whole_hist), axis=0)
        whole_hist[0:hist.shape[0]] += hist
        int_initmin = bin_edges[0]

        assert whole_hist.shape[0] == whole_bin_edges.shape[0] - 1

    elif bin_edges[0] >= int_initmin and bin_edges[-1] > int_initmax:
        # creaet the new store list and copy the old one to it.
        added_bin_edges_right = np.arange(round(int_initmax / interval) + 1,
                                          round(bin_edges[-1] / interval) + 1, 1) * interval
        added_hist_right = np.zeros_like(added_bin_edges_right)
        whole_bin_edges = np.concatenate((whole_bin_edges, added_bin_edges_right), axis=0)
        whole_hist = np.concatenate((whole_hist, added_hist_right), axis=0)
        whole_hist[-hist.shape[0]:] += hist
        int_initmax = bin_edges[-1]
        assert whole_hist.shape[0] == whole_bin_edges.shape[0] - 1

    elif bin_edges[0] < int_initmin and bin_edges[-1] > int_initmax:
        # creaet the new store list and copy the old one to it.
        added_bin_edges_left = np.arange(round(bin_edges[0] / interval), round(int_initmin / interval), 1) * interval
        added_hist_left = np.zeros_like(added_bin_edges_left)
        whole_bin_edges = np.concatenate((added_bin_edges_left, whole_bin_edges), axis=0)
        whole_hist = np.concatenate((added_hist_left, whole_hist), axis=0)
        # whole_hist[0:hist.shape[0]] += hist
        int_initmin = bin_edges[0]

        added_bin_edges_right = np.arange(round(int_initmax / interval) + 1, round(bin_edges[-1] / interval) + 1,
                                          1) * interval
        added_hist_right = np.zeros_like(added_bin_edges_right)
        whole_bin_edges = np.concatenate((whole_bin_edges, added_bin_edges_right), axis=0)
        whole_hist = np.concatenate((whole_hist, added_hist_right), axis=0)
        int_initmax = bin_edges[-1]

        whole_hist += hist
        assert whole_hist.shape[0] == whole_bin_edges.shape[0] - 1

    else:
        pass

    return bin_edges, hist, whole_bin_edges, whole_hist, int_initmin, int_initmax



def main():
    root_dir = '/mnt/sda/3DRadardata/'
    dates = ['2019_04_09', '2019_04_30', '2019_05_09', '2019_05_28', '2019_05_29']
    # dates = ['2019_09_08', '2019_09_18', '2019_09_29']
    # dates = ['2019_09_18', '2019_09_29']
    save_hist_dir = 'hist'
    save_edge_dir = 'bin_edge'

    for capture_date in dates:
        int_initmin = 0
        int_initmax = 0
        cap_folder_dir = os.path.join(root_dir, capture_date)
        seqs = sorted(os.listdir(cap_folder_dir))
        for seq in seqs:
            if norm_pers == 'RA':
                seq_dir = os.path.join(cap_folder_dir, seq, 'RA_NPY')
                chirp_dirs = sorted(os.listdir(seq_dir))
                for fi in range(len(chirp_dirs)):
                    frame_dir = os.path.join(seq_dir, str(fi).zfill(4))
                    npys = sorted(os.listdir(frame_dir))
                    for npy in npys:
                        if random() < thres:
                            npy_dir = os.path.join(frame_dir, npy)
                            data = np.load(npy_dir)
                            real = data.flatten()
                            int_min = (min(real) // interval) * interval
                            int_max = (max(real) // interval + 2) * interval
                            int_bin = np.arange(int_min, int_max, interval)
                            hist, bin_edges = np.histogram(real, bins = int_bin)
                            if int_initmin == 0 and int_initmax == 0:
                                whole_bin_edges = bin_edges
                                whole_hist = hist
                                int_initmin = round(bin_edges[0]/interval) * interval
                                int_initmax = round(bin_edges[-1]/interval) * interval
                            else:
                                bin_edges, hist, whole_bin_edges, whole_hist, int_initmin, int_initmax = \
                                    generate_hist(bin_edges, hist, whole_bin_edges, whole_hist, int_initmin, int_initmax)

                            # print(sum(whole_hist))
                            assert sum(whole_hist) % (32768) == 0

                    print('finish', seq, fi)

            elif norm_pers == 'RV' or norm_pers == 'VA':
                seq_dir = os.path.join(cap_folder_dir, seq, norm_pers) + '_NPY'
                npys = sorted(os.listdir(seq_dir))
                for npy in npys:
                    npy_dir = os.path.join(seq_dir, npy)
                    data = np.load(npy_dir)
                    real = data.flatten()
                    int_min = (min(real) // interval) * interval
                    int_max = (max(real) // interval + 2) * interval
                    int_bin = np.arange(int_min, int_max, interval)
                    hist, bin_edges = np.histogram(real, bins=int_bin)
                    if int_initmin == 0 and int_initmax == 0:
                        whole_bin_edges = bin_edges
                        whole_hist = hist
                        int_initmin = round(bin_edges[0] / interval) * interval
                        int_initmax = round(bin_edges[-1] / interval) * interval
                    else:
                        bin_edges, hist, whole_bin_edges, whole_hist, int_initmin, int_initmax = \
                            generate_hist(bin_edges, hist, whole_bin_edges, whole_hist, int_initmin, int_initmax)

                    # print(sum(whole_hist))
                    print(seq, npy)
                    # assert sum(whole_hist) % (16384) == 0

            print('finished', seq)

        save_edge_dir_seq = save_edge_dir + '_' + capture_date + '_' + norm_pers + '.npy'
        save_hist_dir_seq = save_hist_dir + '_' + capture_date + '_' + norm_pers + '.npy'
        np.save(save_hist_dir_seq, whole_hist)
        np.save(save_edge_dir_seq, whole_bin_edges)


if __name__ == '__main__':
    # main()
    summarize()


