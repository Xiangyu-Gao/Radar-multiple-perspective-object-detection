import numpy as np
import random
import ctypes
import torch
from multiprocessing import Value, Array, Process
from config import radar_configs, rodnet_configs, n_class

ramap_rsize = radar_configs['ramap_rsize']
ramap_asize = radar_configs['ramap_asize']
ramap_vsize = radar_configs['ramap_vsize']


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

        self.win_size = dataset.win_size
        self.shradar_ra = Array(ctypes.c_double, num_parallel_batch * batch_size * 2 * (dataset.win_size*2) * ramap_rsize * ramap_asize)
        self.shradar_rv = Array(ctypes.c_double, num_parallel_batch * batch_size * 1 * (dataset.win_size*2) * ramap_rsize * ramap_vsize)
        self.shradar_va = Array(ctypes.c_double, num_parallel_batch * batch_size * 1 * (dataset.win_size*2) * ramap_asize * ramap_vsize)
        self.shconf = Array(ctypes.c_double, num_parallel_batch * batch_size * n_class * dataset.win_size * ramap_rsize * ramap_asize)
        self.shindex_array = Array(ctypes.c_long, num_parallel_batch * batch_size)
        self.shobj_info_array = Array(ctypes.c_long, 3 * rodnet_configs['max_dets'] * num_parallel_batch * batch_size)  # 3 for obj_info
        self.num_parallel_batch = num_parallel_batch

    def __len__(self):
        return self.length

    def __iter__(self):
        random.shuffle(self.loading_seq)
        procs = [None, None]
        procs[0] = Process(target=self.getBatch, args=(self.shradar_ra, self.shradar_rv, self.shradar_va, self.shconf,
                                                       self.shindex_array, self.shobj_info_array,
                                                       self.loading_seq[0: self.batch_size], 0))
        procs[0].start()

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
                procs[(index_num + 1) % self.num_parallel_batch] = Process(target=self.getBatch, args=(self.shradar_ra,
                    self.shradar_rv, self.shradar_va, self.shconf, self.shindex_array, self.shobj_info_array,
                    loading_seq, (index_num + 1) % self.num_parallel_batch))
                procs[(index_num + 1) % self.num_parallel_batch].start()

            obj_info = self.getBatchObjInfo(i)
            shradarnp_ra = np.frombuffer(self.shradar_ra.get_obj())
            shradarnp_ra = shradarnp_ra.reshape(self.num_parallel_batch, self.batch_size, 2, self.win_size*2, ramap_rsize, ramap_asize)
            shradarnp_rv = np.frombuffer(self.shradar_rv.get_obj())
            shradarnp_rv = shradarnp_rv.reshape(self.num_parallel_batch, self.batch_size, 1, self.win_size*2, ramap_rsize, ramap_vsize)
            shradarnp_va = np.frombuffer(self.shradar_va.get_obj())
            shradarnp_va = shradarnp_va.reshape(self.num_parallel_batch, self.batch_size, 1, self.win_size*2, ramap_asize, ramap_vsize)
            shconfnp = np.frombuffer(self.shconf.get_obj())
            shconfnp = shconfnp.reshape(self.num_parallel_batch, self.batch_size, n_class, self.win_size, ramap_rsize, ramap_asize)
            shindex_arraynp = np.frombuffer(self.shindex_array.get_obj())
            shindex_arraynp = shindex_arraynp.reshape(self.num_parallel_batch, self.batch_size)

            if i < self.length - 1:
                data_length = self.batch_size
            else:
                data_length = len(self.dataset) - self.batch_size * i
            yield torch.from_numpy(shradarnp_ra[index_num, :data_length, : ,: ,:, :]), torch.from_numpy(shradarnp_rv[index_num, :data_length, : ,: ,:, :]), \
                  torch.from_numpy(shradarnp_va[index_num, :data_length, : ,: ,:, :]), torch.from_numpy(shconfnp[index_num, :data_length, : ,: ,:, :]), \
                  obj_info, shindex_arraynp[index_num, :data_length].astype(int)

    def getBatch(self, shradar_ra, shradar_rv, shradar_va, shconf, shindex_array, shobj_info_array, loading_seq, index):
        results = self.dataset.getBatch(loading_seq)
        shradarnp_ra = np.frombuffer(shradar_ra.get_obj())
        shradarnp_ra = shradarnp_ra.reshape(self.num_parallel_batch, self.batch_size, 2, self.win_size*2, ramap_rsize, ramap_asize)
        shradarnp_rv = np.frombuffer(shradar_rv.get_obj())
        shradarnp_rv = shradarnp_rv.reshape(self.num_parallel_batch, self.batch_size, 1, self.win_size*2, ramap_rsize, ramap_vsize)
        shradarnp_va = np.frombuffer(shradar_va.get_obj())
        shradarnp_va = shradarnp_va.reshape(self.num_parallel_batch, self.batch_size, 1, self.win_size*2, ramap_asize, ramap_vsize)
        shconfnp = np.frombuffer(shconf.get_obj())
        shconfnp = shconfnp.reshape(self.num_parallel_batch, self.batch_size, n_class, self.win_size, ramap_rsize, ramap_asize)
        shindex_arraynp = np.frombuffer(shindex_array.get_obj())
        shindex_arraynp = shindex_arraynp.reshape(self.num_parallel_batch, self.batch_size)

        shradarnp_ra[index, :len(loading_seq), :, :, :, :] = results[0]
        shradarnp_rv[index, :len(loading_seq), :, :, :, :] = results[1]
        shradarnp_va[index, :len(loading_seq), :, :, :, :] = results[2]
        shconfnp[index, :len(loading_seq), :, :, :, :] = results[3]
        shindex_arraynp[index, :len(loading_seq)] = results[5]


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
        for i in range(4):
            results[i] = torch.from_numpy(results[i])
        return results
