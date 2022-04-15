import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.read_annotations import read_ra_labels_csv, read_3d_labels_txt
from utils.dataset_tools import fix_cam_drop_frames

from config import train_sets, test_sets
from config import n_class

test_sets_easy = {
    'root_dir': "/mnt/nas_crdataset",
    'dates': ['2019_05_28'],
    'seqs': [
        ['2019_05_28_bm1s011', '2019_05_28_bm1s012', '2019_05_28_bm1s013', '2019_05_28_bm1s014',
         '2019_05_28_cs1s005', '2019_05_28_cs1s006', '2019_05_28_pm2s012', '2019_05_28_pm2s013', '2019_05_28_pm2s014'],
    ],  # 'seqs' is two list corresponding to 'dates', include all seqs if None
    'cam_anno': [False],
}  # test files

test_sets_medium = {
    'root_dir': "/mnt/nas_crdataset",
    'dates': ['2019_05_28'],
    'seqs': [
        ['2019_05_28_bs1s006',
         '2019_05_28_cm1s009', '2019_05_28_cm1s011', '2019_05_28_cm1s012', '2019_05_28_cm1s013',
         '2019_05_28_cs1s004', '2019_05_28_pbms006'],
    ],  # 'seqs' is two list corresponding to 'dates', include all seqs if None
    'cam_anno': [False],
}  # test files

test_sets_hard = {
    'root_dir': "/mnt/nas_crdataset",
    'dates': ['2019_05_28', '2019_09_18'],
    'seqs': [
        ['2019_05_28_cm1s010', '2019_05_28_mlms005',
         '2019_05_28_pcms004', ],
        ['2019_09_18_onrd004', '2019_09_18_onrd009', ],
    ],  # 'seqs' is two list corresponding to 'dates', include all seqs if None
    'cam_anno': [False, True],
}  # test files


def count_objects(obj_info_list):
    counts = np.zeros((n_class,))
    n_frames = np.zeros((n_class))
    ped_hist = []
    cyc_hist = []
    car_hist = []

    for obj_info in obj_info_list:
        flags = np.zeros((n_class))
        counts_in_frame = np.zeros((n_class))
        for obj in obj_info:
            class_id = obj[2]
            if class_id >= n_class or class_id < 0:
                continue
            counts[class_id] += 1
            counts_in_frame[class_id] += 1
            flags[class_id] = 1
        n_frames += flags
        if counts_in_frame[0] != 0:
            ped_hist.append(counts_in_frame[0])
        if counts_in_frame[1] != 0:
            cyc_hist.append(counts_in_frame[1])
        if counts_in_frame[2] != 0:
            car_hist.append(counts_in_frame[2])
    return counts, n_frames, [ped_hist, cyc_hist, car_hist]


def draw_figures(counts, n_frames):
    # df = pd.DataFrame({
    #     'Class': ['Pedestrian', 'Cyclist', 'Car'],
    #     'Training': [counts[0][0], counts[0][1], counts[0][2]],
    #     'Testing': [counts[1][0], counts[1][1], counts[1][2]],
    # })
    # df.groupby(['Class', 'Training']).size().unstack().plot(kind='bar', stacked=True)

    df = pd.DataFrame([['Pedestrian', 'Training', counts[0][0]], ['Pedestrian', 'Testing', counts[1][0]],
                       ['Cyclist', 'Training', counts[0][1]], ['Cyclist', 'Testing', counts[1][1]],
                       ['Car', 'Training', counts[0][2]], ['Car', 'Testing', counts[1][2]]],
                      columns=['Class', 'Dataset', 'Number of Objects'])
    df.pivot("Dataset", "Class", "Number of Objects").plot(kind='bar')
    plt.show()


def draw_hists(hists):
    for stats in hists:
        hist, bin_edges = np.histogram(stats)
        n, bins, patches = plt.hist(x=stats, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('My Very Own Histogram')
        plt.text(23, 45, r'$\mu=15, b=3$')
        maxfreq = n.max()
        # Set a clean upper y-axis limit.
        plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        plt.show()


if __name__ == '__main__':
    if True:
        datasets = [test_sets]
        # datasets = [test_sets_easy, test_sets_medium, test_sets_hard]
        counts_all = []
        n_frames_all = []

        for dataset in datasets:
            root_dir = dataset['root_dir']
            dates = dataset['dates']
            seqs = dataset['seqs']
            cam_anno = dataset['cam_anno']

            obj_info_list = []
            for date_id, date in enumerate(dates):
                print('loading annotations for %s' % os.path.join(root_dir, date))
                this_seqs = seqs[date_id]
                this_cam_anno = cam_anno[date_id]
                if this_seqs is None:
                    this_seqs = sorted(os.listdir(os.path.join(root_dir, date)))
                for seq in this_seqs:
                    seq_path = os.path.join(root_dir, date, seq)
                    if this_cam_anno:
                        # use camera annotations
                        label_names = sorted(os.listdir(os.path.join(seq_path, 'dets_3d')))
                        n_labels = len(label_names)  # number of label files
                        label_names = fix_cam_drop_frames(seq_path, label_names)

                        for label_id in range(n_labels):
                            # for each frame
                            label_name = label_names[label_id]
                            obj_info = read_3d_labels_txt(seq_path, label_name, 'dets_3d')
                            obj_info_list.append(obj_info)
                            # end objects loop
                    else:
                        # use labelled RAMap
                        try:
                            obj_info_cur = read_ra_labels_csv(seq_path)
                            obj_info_list.extend(obj_info_cur)
                        except Exception as e:
                            print("Load sequence %s failed!" % seq_path)
                            print(e)
                            continue

            counts, n_frames, hists = count_objects(obj_info_list)
            counts_all.append(counts)
            n_frames_all.append(n_frames)
    else:
        counts_all = [np.array([64414., 28907., 146008.]), np.array([9465., 5677., 7859.])]
        n_frames_all = [np.array([41478., 28545., 61902.]), np.array([5965., 5674., 6047.])]

    print(counts_all)
    print(n_frames_all)

    # draw_figures(counts_all, n_frames_all)
    # draw_hists(hists)
