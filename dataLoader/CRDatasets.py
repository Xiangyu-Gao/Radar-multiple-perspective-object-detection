import os
import time
import random
import pickle
import numpy as np

from torch.utils import data
from config import radar_configs, rodnet_configs, n_class


class CRDataset(data.Dataset):
    """
    Pytorch Dataloader for CR Dataset
    :param detail_dir: data details directory
    :param confmap_dir: confidence maps directory
    :param win_size: seqence window size
    :param n_class: number of classes for detection
    :param step: frame step inside each sequence
    :param stride: data sampling
    :param set_type: train, valid, test
    :param is_random: random load or not
    """
    def __init__(self, detail_dir, confmap_dir=None, win_size=16, n_class=3, step=1, stride=1,
                 set_type='train', is_random=True, subset=None, noise_channel=False):
        if set_type == 'train' or set_type == 'valid':
            assert confmap_dir is not None
        self.labels = []
        self.confmaps = []
        self.win_size = win_size
        self.n_class = n_class
        self.step = step
        self.stride = stride
        self.noise_channel = noise_channel
        if subset != None:
            detail_files = [subset + '.pkl']
        else:
            detail_files = sorted(os.listdir(os.path.join(detail_dir, set_type)))
        if confmap_dir is not None:
            if subset != None:
                confmap_files = [subset + '.pkl']
            else:
                confmap_files = sorted(os.listdir(os.path.join(confmap_dir, set_type)))
            assert len(detail_files) == len(confmap_files)
        self.seq_names = [name.split('.')[0] for name in detail_files]
        self.n_seq = len(self.seq_names)
        self.n_label = 0
        self.index_mapping = []
        for seq_id, detail_file in enumerate(detail_files):
            data_details = pickle.load(open(os.path.join(detail_dir, set_type, detail_file), 'rb'))
            self.labels.append(data_details[0])
            n_data_in_seq = (len(self.labels[-1]) - (self.win_size * step - 1)) // stride \
                                + (1 if (len(self.labels[-1]) - (self.win_size * step - 1)) % stride > 0 else 0)
            self.n_label += n_data_in_seq
            for data_id in range(n_data_in_seq):
                self.index_mapping.append([seq_id, data_id * stride, data_details[1]])
        if confmap_dir is not None:
            for confmap_file in confmap_files:
                confmaps = pickle.load(open(os.path.join(confmap_dir, set_type, confmap_file), 'rb'))
                self.confmaps.append(confmaps)
        self.set_type = set_type
        self.is_random = is_random

    def __len__(self):
        """Total number of data/label pairs"""
        return self.n_label

    def __getitem__(self, index):

        seq_id, data_id, ra_frame_offset = self.index_mapping[index]
        this_label = self.labels[seq_id][data_id]
        path = this_label
        if self.confmaps != []:
            this_seq_confmap = self.confmaps[seq_id]

        if self.is_random:
            if self.seq_names[seq_id][0:10] == '2020_00_00':
                chirp_id = 0
                chirp_id2 = 128
            else:
                chirp_id = random.randint(0, int((radar_configs['n_chirps'] - 1) / 2))
                chirp_id2 = chirp_id + int((radar_configs['n_chirps'] - 1) / 2)
        else:
            chirp_id = 0

        ramap_rsize = radar_configs['ramap_rsize']
        ramap_asize = radar_configs['ramap_asize']
        ramap_vsize = radar_configs['ramap_vsize']

        try:
            if radar_configs['data_type'] == 'RI' or radar_configs['data_type'] == 'AP':
                radar_npy_win = np.load(path % (chirp_id)) \
                    [ra_frame_offset + data_id:ra_frame_offset + data_id + self.win_size * self.step:self.step, :, :, :]
            elif radar_configs['data_type'] == 'RISEP' or radar_configs['data_type'] == 'APSEP':
                radar_npy_win_ra = np.zeros((self.win_size * 2, ramap_rsize, ramap_asize, 2), dtype=np.float32)
                radar_npy_win_rv = np.zeros((self.win_size * 2, ramap_rsize, ramap_vsize, 1), dtype=np.float32)
                radar_npy_win_va = np.zeros((self.win_size * 2, ramap_asize, ramap_vsize, 1), dtype=np.float32)
                root_path = path.split('/RA_NPY')[0]
                for idx, frameid in enumerate(range(ra_frame_offset + data_id, ra_frame_offset + data_id + self.win_size * self.step, self.step)):
                    # load ra slice
                    # format of radar_npy_win_ra [chirp, range, angle, real/imag]
                    radar_npy_win_ra[idx * 2, :, :, :] = np.load(path % (chirp_id, frameid))
                    radar_npy_win_ra[idx * 2 + 1, :, :, :] = np.load(path % (chirp_id2, frameid))
                    # load rv slice
                    # format of radar_npy_win_rv [chirp, range, velocity, real]
                    path_rv = root_path + '/RV_NPY/' + str(frameid).zfill(6) + '.npy'
                    load_rv = np.load(path_rv)
                    radar_npy_win_rv[idx * 2, :, :, 0] = load_rv[:, :, 0]
                    radar_npy_win_rv[idx * 2 + 1, :, :, 0] = load_rv[:, :, 1]
                    # load va slice
                    # format of radar_npy_win_rv [chirp, angle, velocity, real]
                    path_va = root_path + '/VA_NPY/' + str(frameid).zfill(6) + '.npy'
                    load_va = np.load(path_va)
                    radar_npy_win_va[idx * 2, :, :, 0] = load_va[:, :, 0]
                    radar_npy_win_va[idx * 2 + 1, :, :, 0] = load_va[:, :, 1]
            else:
                raise ValueError
        except:
            print('load fail')
            raise ValueError
            # # in case load npy fail
            # if not os.path.exists('./tmp'):
            #     os.makedirs('./tmp')
            # log_name = 'loadnpyfail-' + time.strftime("%Y%m%d-%H%M%S") + '.txt'
            # with open(os.path.join('./tmp', log_name), 'w') as f_log:
            #     f_log.write('npy path: ' + path % (chirp_id, frameid) + \
            #                 '\nframe indices: %d:%d:%d' % (data_id, data_id + self.win_size * self.step, self.step))
            # radar_npy_win = np.transpose(radar_npy_win, (3, 0, 1, 2))
            # if self.confmaps != []:
            #     confmap_gt = this_seq_confmap[0][data_id:data_id + self.win_size * self.step:self.step]
            #     confmap_gt = np.transpose(confmap_gt, (1, 0, 2, 3))
            #     obj_info = this_seq_confmap[1][data_id:data_id + self.win_size * self.step:self.step]
            #     return radar_npy_win, confmap_gt, obj_info, -1
            # else:
            #     return radar_npy_win, -1

        radar_npy_win_ra = np.transpose(radar_npy_win_ra, (3, 0, 1, 2))
        radar_npy_win_rv = np.transpose(radar_npy_win_rv, (3, 0, 1, 2))
        radar_npy_win_va = np.transpose(radar_npy_win_va, (3, 0, 1, 2))
        assert radar_npy_win_ra.shape == (2, self.win_size * 2, ramap_rsize, ramap_asize)
        assert radar_npy_win_rv.shape == (1, self.win_size * 2, ramap_rsize, ramap_vsize)
        assert radar_npy_win_va.shape == (1, self.win_size * 2, ramap_asize, ramap_vsize)

        if self.confmaps != []:
            confmap_gt = this_seq_confmap[0][data_id:data_id + self.win_size * self.step:self.step]
            confmap_gt = np.transpose(confmap_gt, (1, 0, 2, 3))
            obj_info = this_seq_confmap[1][data_id:data_id + self.win_size * self.step:self.step]
            if self.noise_channel:
                assert confmap_gt.shape == \
                       (n_class + 1, self.win_size, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])
            else:
                confmap_gt = confmap_gt[:n_class]
                assert confmap_gt.shape == \
                       (n_class, self.win_size, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])
                assert np.shape(obj_info)[0] == self.win_size
            return radar_npy_win_ra, radar_npy_win_rv, radar_npy_win_va, confmap_gt, obj_info, index

        else:
            return radar_npy_win_ra, radar_npy_win_rv, radar_npy_win_va, index


class CRDatasetSM(data.Dataset):
    """
    Pytorch Dataloader for CR Dataset
    :param detail_dir: data details directory
    :param confmap_dir: confidence maps directory
    :param win_size: sequence window size
    :param n_class: number of classes for detection
    :param step: frame step inside each sequence
    :param stride: data sampling
    :param set_type: train, valid, test
    :param is_random: random load or not
    """
    def __init__(self, detail_dir, confmap_dir=None, win_size=16, n_class=3, step=1, stride=1,
                 set_type='train', is_random=True, subset=None, noise_channel=False,
                 is_Memory_Limit=True):
        if set_type == 'train':
            assert confmap_dir is not None
        self.labels = []
        self.confmaps = []
        self.win_size = win_size
        self.n_class = n_class
        self.step = step
        self.stride = stride
        self.noise_channel = noise_channel
        if subset != None:
            detail_files = [subset + '.pkl']
        else:
            detail_files = sorted(os.listdir(os.path.join(detail_dir, set_type)))
        if confmap_dir is not None:
            if subset != None:
                confmap_files = [subset + '.pkl']
            else:
                confmap_files = sorted(os.listdir(os.path.join(confmap_dir, set_type)))
            assert len(detail_files) == len(confmap_files)
        self.seq_names = [name.split('.')[0] for name in detail_files]
        self.n_seq = len(self.seq_names)
        self.n_label = 0
        self.index_mapping = []
        for seq_id, detail_file in enumerate(detail_files):
            data_details = pickle.load(open(os.path.join(detail_dir, set_type, detail_file), 'rb'))
            self.labels.append(data_details[0])
            n_data_in_seq = (len(self.labels[-1]) - (self.win_size * step - 1)) // stride \
                                + (1 if (len(self.labels[-1]) - (self.win_size * step - 1)) % stride > 0 else 0)
            self.n_label += n_data_in_seq
            for data_id in range(n_data_in_seq):
                self.index_mapping.append([seq_id, data_id * stride, data_details[1]])
        self.is_Memory_Limit = is_Memory_Limit
        if confmap_dir is not None and not is_Memory_Limit:
            for confmap_file in confmap_files:
                confmaps = pickle.load(open(os.path.join(confmap_dir, set_type, confmap_file), 'rb'))
                self.confmaps.append(confmaps)
        elif confmap_dir is not None and is_Memory_Limit:
            self.confmap_dir = confmap_dir
            self.confmap_files = confmap_files
            self.set_type = set_type
        elif is_Memory_Limit:
            self.confmap_files = None
        self.set_type = set_type
        self.is_random = is_random

    def __len__(self):
        """Total number of data/label pairs"""
        return self.n_label

    def getObjInfo(self, index):
        seq_id, data_id, ra_frame_offset = self.index_mapping[index]
        if self.confmaps != [] and not self.is_Memory_Limit:
            this_seq_confmap = self.confmaps[seq_id]
        elif self.is_Memory_Limit and self.confmap_files is not None:
            this_seq_confmap = pickle.load(open(os.path.join(self.confmap_dir, self.set_type, self.confmap_files[seq_id]), 'rb'))
        obj_info = this_seq_confmap[1][ra_frame_offset + data_id:ra_frame_offset + data_id + self.win_size * self.step:self.step]
        return obj_info

    def getBatchObjInfo(self, indexes):
        bObj_info = []
        for index in indexes:
            result = self.getObjInfo(index)
            bObj_info.append(result)
        return bObj_info

    def getBatch(self, indexes):
        bRadar_npy_win_ra = []
        bRadar_npy_win_rv = []
        bRadar_npy_win_va = []
        bConfmap_gt = []
        bObj_info = []
        for index in indexes:
            results = self.__getitem__(index)
            bRadar_npy_win_ra.append(results[0])
            bRadar_npy_win_rv.append(results[1])
            bRadar_npy_win_va.append(results[2])
            bConfmap_gt.append(results[3])
            bObj_info.append(results[4])
        return np.array(bRadar_npy_win_ra, dtype=np.float32), np.array(bRadar_npy_win_rv, dtype=np.float32), \
               np.array(bRadar_npy_win_va, dtype=np.float32), np.array(bConfmap_gt, dtype=np.float32), \
               bObj_info, np.array(indexes, dtype=np.int32)


    def __getitem__(self, index):
        seq_id, data_id, ra_frame_offset = self.index_mapping[index]
        this_label = self.labels[seq_id][data_id]
        path = this_label
        if self.confmaps != [] and not self.is_Memory_Limit:
            this_seq_confmap = self.confmaps[seq_id]
        elif self.is_Memory_Limit and self.confmap_files is not None:
            this_seq_confmap = pickle.load(open(os.path.join(self.confmap_dir, self.set_type, self.confmap_files[seq_id]), 'rb'))

        if self.is_random:
            if self.seq_names[seq_id][0:10] == '2020_00_00':
                chirp_id = 0
                chirp_id2 = 128
            else:
                chirp_id = random.randint(0, int((radar_configs['n_chirps']-1)/2))
                chirp_id2 = chirp_id + int((radar_configs['n_chirps']-1)/2)
        else:
            chirp_id = 0
            chirp_id2 = 128

        ramap_rsize = radar_configs['ramap_rsize']
        ramap_asize = radar_configs['ramap_asize']
        ramap_vsize = radar_configs['ramap_vsize']

        try:
            if radar_configs['data_type'] == 'RI' or radar_configs['data_type'] == 'AP':
                radar_npy_win = np.load(path % (chirp_id)) \
                    [ra_frame_offset + data_id:ra_frame_offset + data_id + self.win_size * self.step:self.step, :, :, :]
            elif radar_configs['data_type'] == 'RISEP' or radar_configs['data_type'] == 'APSEP':
                radar_npy_win_ra = np.zeros((self.win_size * 2, ramap_rsize, ramap_asize, 2), dtype=np.float32)
                radar_npy_win_rv = np.zeros((self.win_size * 2, ramap_rsize, ramap_vsize, 1), dtype=np.float32)
                radar_npy_win_va = np.zeros((self.win_size * 2, ramap_asize, ramap_vsize, 1), dtype=np.float32)
                root_path = path.split('/RA_NPY')[0]
                for idx, frameid in enumerate(range(ra_frame_offset + data_id, ra_frame_offset + data_id + self.win_size * self.step, self.step)):
                    # load ra slice
                    # format of radar_npy_win_ra [chirp, range, angle, real/imag]
                    radar_npy_win_ra[idx * 2, :, :, :] = np.load(path % (chirp_id, frameid))
                    radar_npy_win_ra[idx * 2 + 1, :, :, :] = np.load(path % (chirp_id2, frameid))
                    # load rv slice
                    # format of radar_npy_win_rv [chirp, range, velocity, real]
                    path_rv = root_path + '/RV_NPY/' + str(frameid).zfill(6) + '.npy'
                    load_rv = np.load(path_rv)
                    radar_npy_win_rv[idx * 2, :, :, 0] = load_rv[:, :, 0]
                    radar_npy_win_rv[idx * 2 + 1, :, :, 0] = load_rv[:, :, 1]
                    # load va slice
                    # format of radar_npy_win_rv [chirp, angle, velocity, real]
                    path_va = root_path + '/VA_NPY/' + str(frameid).zfill(6) + '.npy'
                    load_va = np.load(path_va)
                    radar_npy_win_va[idx * 2, :, :, 0] = load_va[:, :, 0]
                    radar_npy_win_va[idx * 2 + 1, :, :, 0] = load_va[:, :, 1]
            else:
                raise ValueError
        except:
            print('load fail')
            raise ValueError
            # # in case load npy fail
            # if not os.path.exists('./tmp'):
            #     os.makedirs('./tmp')
            # log_name = 'loadnpyfail-' + time.strftime("%Y%m%d-%H%M%S") + '.txt'
            # with open(os.path.join('./tmp', log_name), 'w') as f_log:
            #     f_log.write('npy path: ' + path % (chirp_id, frameid) + \
            #                 '\nframe indices: %d:%d:%d' % (data_id, data_id + self.win_size * self.step, self.step))
            # radar_npy_win = np.transpose(radar_npy_win, (3, 0, 1, 2))
            # if self.confmaps != []:
            #     confmap_gt = this_seq_confmap[0][data_id:data_id + self.win_size * self.step:self.step]
            #     confmap_gt = np.transpose(confmap_gt, (1, 0, 2, 3))
            #     obj_info = this_seq_confmap[1][data_id:data_id + self.win_size * self.step:self.step]
            #     return radar_npy_win, confmap_gt, obj_info, -1
            # else:
            #     return radar_npy_win, -1

        radar_npy_win_ra = np.transpose(radar_npy_win_ra, (3, 0, 1, 2))
        radar_npy_win_rv = np.transpose(radar_npy_win_rv, (3, 0, 1, 2))
        radar_npy_win_va = np.transpose(radar_npy_win_va, (3, 0, 1, 2))
        assert radar_npy_win_ra.shape == (2, self.win_size * 2, ramap_rsize, ramap_asize)
        assert radar_npy_win_rv.shape == (1, self.win_size * 2, ramap_rsize, ramap_vsize)
        assert radar_npy_win_va.shape == (1, self.win_size * 2, ramap_asize, ramap_vsize)

        if (self.confmaps != [] and not self.is_Memory_Limit) or (self.is_Memory_Limit and self.confmap_files is not None):
            confmap_gt = this_seq_confmap[0][data_id:data_id + self.win_size * self.step:self.step]
            confmap_gt = np.transpose(confmap_gt, (1, 0, 2, 3))
            obj_info = this_seq_confmap[1][data_id:data_id + self.win_size * self.step:self.step]
            if self.noise_channel:
                assert confmap_gt.shape == \
                       (n_class + 1, self.win_size, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])
            else:
                confmap_gt = confmap_gt[:n_class]
                assert confmap_gt.shape == \
                       (n_class, self.win_size, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])
                assert np.shape(obj_info)[0] == self.win_size

            return radar_npy_win_ra, radar_npy_win_rv, radar_npy_win_va, confmap_gt, obj_info, index

        else:
            return radar_npy_win_ra, radar_npy_win_rv, radar_npy_win_va, index


if __name__ == "__main__":
    dataset = CRDatasetSM('./data/data_details', './data/confmaps_gt', stride=16)
    print(len(dataset))
    for i in range(len(dataset)):
        radar_npy_win_ra, radar_npy_win_rv, radar_npy_win_va, confmap_gt, obj_info, index = dataset.__getitem__(i)
        print(radar_npy_win_ra.shape)
        print(radar_npy_win_rv.shape)
        print(radar_npy_win_va.shape)
        print(confmap_gt.shape)
        input()
        continue
