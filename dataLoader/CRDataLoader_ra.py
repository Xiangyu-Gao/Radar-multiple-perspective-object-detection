import numpy as np
import random
import ctypes
from multiprocessing import Value, Array, Process

import torch

from config import radar_configs, rodnet_configs, n_class


class CRDataLoader():
    def __init__(self, dataset, batch_size=1, shuffle=False, num_parallel_batch=2):
        self.batch_size = batch_size
        self.length = len(dataset) // self.batch_size + (1 if len(dataset) % self.batch_size != 0 else 0)
        self.loading_seq = [i for i in range(len(dataset))]
        self.dataset = dataset
        if shuffle:
            random.shuffle(self.loading_seq)
        self.restart = False

        assert num_parallel_batch > 0 and type(num_parallel_batch) == int
        ramap_rsize = radar_configs['ramap_rsize']
        ramap_asize = radar_configs['ramap_asize']

        self.win_size = dataset.win_size
        self.shradar = Array(ctypes.c_double, num_parallel_batch * batch_size * 2 * dataset.win_size * ramap_rsize * ramap_asize)
        self.shconf = Array(ctypes.c_double, num_parallel_batch * batch_size * n_class * dataset.win_size * ramap_rsize * ramap_asize)
        self.shindex_array = Array(ctypes.c_long, num_parallel_batch * batch_size)
        self.shobj_info_array = Array(ctypes.c_long, 3 * rodnet_configs['max_dets'] * num_parallel_batch * batch_size)  # 3 for obj_info
        self.num_parallel_batch = num_parallel_batch

    def __len__(self):
        return self.length

    def __iter__(self):
        random.shuffle(self.loading_seq)
        procs = [None, None]
        procs[0] = Process(target=self.getBatch, args=(self.shradar, self.shconf, self.shindex_array, self.shobj_info_array,
                                self.loading_seq[0: self.batch_size], 0))
        procs[0].start()
        ramap_rsize = radar_configs['ramap_rsize']
        ramap_asize = radar_configs['ramap_asize']

        index_num = self.num_parallel_batch - 1
        for i in range(self.__len__()):
            index_num = (index_num + 1) % self.num_parallel_batch
            procs[index_num].join()
            procs[index_num] = None

            if i < self.length - self.num_parallel_batch:
                loading_seq = self.loading_seq[self.batch_size * (i + self.num_parallel_batch - 1): self.batch_size * (i + self.num_parallel_batch)]
            else:
                loading_seq = self.loading_seq[self.batch_size * (i + self.num_parallel_batch - 1):]

            if i < self.length - self.num_parallel_batch + 1:
                procs[(index_num + 1) % self.num_parallel_batch] = Process(target=self.getBatch, args=(self.shradar, self.shconf, self.shindex_array, self.shobj_info_array,
                                            loading_seq, (index_num + 1) % self.num_parallel_batch))
                procs[(index_num + 1) % self.num_parallel_batch].start()

            obj_info = self.getBatchObjInfo(i)
            shradarnp = np.frombuffer(self.shradar.get_obj())
            shradarnp = shradarnp.reshape(self.num_parallel_batch, self.batch_size, 2, self.win_size, ramap_rsize, ramap_asize)
            shconfnp = np.frombuffer(self.shconf.get_obj())
            shconfnp = shconfnp.reshape(self.num_parallel_batch, self.batch_size, n_class, self.win_size, ramap_rsize, ramap_asize)
            shindex_arraynp = np.frombuffer(self.shindex_array.get_obj())
            shindex_arraynp = shindex_arraynp.reshape(self.num_parallel_batch, self.batch_size)

            if i < self.length - 1:
                data_length = self.batch_size
            else:
                data_length = len(self.dataset) - self.batch_size * i
            yield torch.from_numpy(shradarnp[index_num, :data_length, : ,: ,:, :]), torch.from_numpy(shconfnp[index_num, :data_length, : ,: ,:, :]), \
                        obj_info, shindex_arraynp[index_num, :data_length].astype(int)

    def getBatch(self, shradar, shconf, shindex_array, shobj_info_array, loading_seq, index):
        ramap_rsize = radar_configs['ramap_rsize']
        ramap_asize = radar_configs['ramap_asize']

        results = self.dataset.getBatch(loading_seq)
        shradarnp = np.frombuffer(shradar.get_obj())
        shradarnp = shradarnp.reshape(self.num_parallel_batch, self.batch_size, 2, self.win_size, ramap_rsize, ramap_asize)
        shconfnp = np.frombuffer(shconf.get_obj())
        shconfnp = shconfnp.reshape(self.num_parallel_batch, self.batch_size, n_class, self.win_size, ramap_rsize, ramap_asize)
        shindex_arraynp = np.frombuffer(shindex_array.get_obj())
        shindex_arraynp = shindex_arraynp.reshape(self.num_parallel_batch, self.batch_size)

        shradarnp[index, :len(loading_seq), :, :, :, :] = results[0]
        shconfnp[index, :len(loading_seq), :, :, :, :] = results[1]
        shindex_arraynp[index, :len(loading_seq)] = results[3]

    def getBatchObjInfo(self, index):
        if index == self.length - 1:
            results = self.dataset.getBatchObjInfo(self.loading_seq[self.batch_size * index:])
        else:
            results = self.dataset.getBatchObjInfo(self.loading_seq[self.batch_size * index: self.batch_size * (index + 1)])
        return results

    def __getitem__(self, index):
        if self.restart:
            random.shuffle(self.loading_seq)
        if index == self.length - 1:
            self.restart = True
            results = self.dataset.getBatch(self.loading_seq[self.batch_size * index:])
        else:
            results = self.dataset.getBatch(self.loading_seq[self.batch_size * index: self.batch_size * (index + 1)])
        results = list(results)
        for i in range(2):
            results[i] = torch.from_numpy(results[i])
        return results
