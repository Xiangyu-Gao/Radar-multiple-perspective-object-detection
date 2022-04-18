from torch.utils.data import DataLoader

from dataLoader.CRDatasets import CRDataset

from config import rodnet_configs


if __name__ == "__main__":
    win_size = rodnet_configs['win_size']
    crdata_train = CRDataset('./data/data_details', './data/confmaps_gt', win_size=win_size, set_type='valid')
    seq_names = crdata_train.seq_names
    index_mapping = crdata_train.index_mapping
    dataloader = DataLoader(crdata_train, batch_size=1, shuffle=True, num_workers=0)

    for iter, (data, confmap_gt, obj_info, real_id) in enumerate(dataloader):
        print(confmap_gt.shape)

