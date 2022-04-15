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
            else:
                chirp_id = random.randint(0, radar_configs['n_chirps']-1)
        else:
            chirp_id = 0

        ramap_rsize = radar_configs['ramap_rsize']
        ramap_asize = radar_configs['ramap_asize']

        try:
            if radar_configs['data_type'] == 'RI' or radar_configs['data_type'] == 'AP':
                radar_npy_win = np.load(path % (chirp_id)) \
                    [ra_frame_offset + data_id:ra_frame_offset + data_id + self.win_size * self.step:self.step, :, :, :]
            elif radar_configs['data_type'] == 'RISEP' or radar_configs['data_type'] == 'APSEP':
                radar_npy_win = np.zeros((self.win_size, ramap_rsize, ramap_asize, 2), dtype=np.float32)
                for idx, frameid in enumerate(range(ra_frame_offset + data_id, ra_frame_offset + data_id + self.win_size * self.step, self.step)):
                    radar_npy_win[idx, :, :, :] = np.load(path % (chirp_id, frameid))
            else:
                raise ValueError
        except:
            # in case load npy fail
            if not os.path.exists('../tmp'):
                os.makedirs('../tmp')
            log_name = 'loadnpyfail-' + time.strftime("%Y%m%d-%H%M%S") + '.txt'
            with open(os.path.join('../tmp', log_name), 'w') as f_log:
                f_log.write('npy path: ' + path % (chirp_id, frameid) + \
                            '\nframe indices: %d:%d:%d' % (data_id, data_id + self.win_size * self.step, self.step))
            radar_npy_win = np.transpose(radar_npy_win, (3, 0, 1, 2))
            if self.confmaps != []:
                confmap_gt = this_seq_confmap[0][data_id:data_id + self.win_size * self.step:self.step]
                confmap_gt = np.transpose(confmap_gt, (1, 0, 2, 3))
                obj_info = this_seq_confmap[1][data_id:data_id + self.win_size * self.step:self.step]
                return radar_npy_win, confmap_gt, obj_info, -1
            else:
                return radar_npy_win, -1
        radar_npy_win = np.transpose(radar_npy_win, (3, 0, 1, 2))
        assert radar_npy_win.shape == (2, self.win_size, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

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

            return radar_npy_win, confmap_gt, obj_info, index

        else:
            return radar_npy_win, index


class CRDatasetSM(data.Dataset):
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
        bRadar_npy_win = []
        bConfmap_gt = []
        bObj_info = []
        for index in indexes:
            results = self.__getitem__(index)
            bRadar_npy_win.append(results[0])
            bConfmap_gt.append(results[1])
            bObj_info.append(results[2])
        return np.array(bRadar_npy_win, dtype=np.float32), np.array(bConfmap_gt, dtype=np.float32), bObj_info, np.array(indexes, dtype=np.int32)


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
            else:
                chirp_id = random.randint(0, radar_configs['n_chirps']-1)
        else:
            chirp_id = 0

        ramap_rsize = radar_configs['ramap_rsize']
        ramap_asize = radar_configs['ramap_asize']

        try:
            if radar_configs['data_type'] == 'RI' or radar_configs['data_type'] == 'AP':
                radar_npy_win = np.load(path % (chirp_id)) \
                    [ra_frame_offset + data_id:ra_frame_offset + data_id + self.win_size * self.step:self.step, :, :, :]
            elif radar_configs['data_type'] == 'RISEP' or radar_configs['data_type'] == 'APSEP':
                radar_npy_win = np.zeros((self.win_size, ramap_rsize, ramap_asize, 2), dtype=np.float32)
                for idx, frameid in enumerate(range(ra_frame_offset + data_id, ra_frame_offset + data_id + self.win_size * self.step, self.step)):
                    radar_npy_win[idx, :, :, :] = np.load(path % (chirp_id, frameid))
            else:
                raise ValueError
        except:
            # in case load npy fail
            if not os.path.exists('../tmp'):
                os.makedirs('../tmp')
            log_name = 'loadnpyfail-' + time.strftime("%Y%m%d-%H%M%S") + '.txt'
            with open(os.path.join('../tmp', log_name), 'w') as f_log:
                f_log.write('npy path: ' + path % (chirp_id, frameid) + \
                            '\nframe indices: %d:%d:%d' % (data_id, data_id + self.win_size * self.step, self.step))
            radar_npy_win = np.transpose(radar_npy_win, (3, 0, 1, 2))
            if self.confmaps != []:
                confmap_gt = this_seq_confmap[0][data_id:data_id + self.win_size * self.step:self.step]
                confmap_gt = np.transpose(confmap_gt, (1, 0, 2, 3))
                obj_info = this_seq_confmap[1][data_id:data_id + self.win_size * self.step:self.step]
                return radar_npy_win, confmap_gt, obj_info, -1
            else:
                return radar_npy_win, -1
        radar_npy_win = np.transpose(radar_npy_win, (3, 0, 1, 2))
        assert radar_npy_win.shape == (2, self.win_size, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

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

            return radar_npy_win, confmap_gt, obj_info, index

        else:
            return radar_npy_win, index


if __name__ == "__main__":
    dataset = CRDataset('./data/data_details', './data/confmaps_gt', stride=16)
    print(dataset.index_mapping[0])
    input()
    print(len(dataset))
    for i in range(len(dataset)):
        continue
