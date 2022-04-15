import os
import time
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from dataLoader.CRDatasets import CRDataset, CRDatasetSM
from dataLoader.CRDataLoader import CRDataLoader
from utils import chirp_amp
from utils.visualization import visualize_train_img

from config import n_class, train_sets
from config import camera_configs, radar_configs, rodnet_configs


def parse_args():
    parser = argparse.ArgumentParser(description='Train RODNet.')
    parser.add_argument('-m', '--model', type=str, dest='model',
                        help='choose rodnet model')
    parser.add_argument('-md', '--modeldir', type=str, dest='model_dir',
                        help='file name to save trained model')
    parser.add_argument('-dd', '--datadir', type=str, dest='data_dir', default='./data/',
                        help='directory to load data')
    parser.add_argument('-ld', '--logdir', type=str, dest='log_dir', default='./results/',
                        help='directory to save trained model')
    parser.add_argument('-sm', '--save_memory', action="store_true", help="use customized dataloader to save memory")
    args = parser.parse_args()
    return args


def create_dir_for_new_model(name='rodnet'):
    model_name = name + '-' + time.strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(os.path.join(train_model_path, model_name)):
        os.makedirs(os.path.join(train_model_path, model_name))
    return model_name


if __name__ == "__main__":
    """
    Example:
        python train.py -m HG -dd /mnt/ssd2/rodnet/data_refine/ -ld /mnt/ssd2/rodnet/checkpoints/ \
            -sm -md HG-20200122-104604
    """
    args = parse_args()

    if args.model == 'CDC':
        from model.RODNet_CDC import RODNet
    elif args.model == 'HG':
        from model.RODNet_HG import RODNet
    elif args.model == 'HGwI':
        from model.RODNet_HGwI import RODNet
    else:
        raise TypeError

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    train_model_path = args.log_dir
    # train_model_path = os.path.join(args.log_dir, 'trained_model')
    # if not os.path.exists(train_model_path):
    #     os.makedirs(train_model_path)

    # create / load models
    model_name = None
    epoch_start = 0
    iter_start = 0
    if args.model_dir is not None and os.path.exists(os.path.join(train_model_path, args.model_dir)):
        model_dir = args.model_dir
        models_loaded = sorted(os.listdir(os.path.join(train_model_path, model_dir)))
        for fid, file in enumerate(models_loaded):
            if not file.endswith('.pkl'):
                del models_loaded[fid]
        if len(models_loaded) == 0:
            model_dir = create_dir_for_new_model()
        else:
            model_name = models_loaded[-1]
            epoch_start = int(float(model_name.split('.')[0].split('_')[1]))
            iter_start = int(float(model_name.split('.')[0].split('_')[2]))
    else:
        model_dir = create_dir_for_new_model(name=args.model)

    train_viz_path = os.path.join(train_model_path, model_dir, 'train_viz')
    if not os.path.exists(train_viz_path):
        os.makedirs(train_viz_path)
    # train_viz_dir = os.path.join(train_viz_path, model_dir)
    # if not os.path.exists(train_viz_dir):
    #     os.makedirs(train_viz_dir)

    writer = SummaryWriter(os.path.join(train_model_path, model_dir))
    save_config_dict = {
        'args': vars(args),
        'camera': camera_configs,
        'radar': radar_configs,
        'rodnet': rodnet_configs,
        'trainset': train_sets,
    }
    config_json_name = os.path.join(train_model_path, model_dir, 'config-'+time.strftime("%Y%m%d-%H%M%S")+'.json')
    with open(config_json_name, 'w') as fp:
        json.dump(save_config_dict, fp)

    n_epoch = rodnet_configs['n_epoch']
    win_size = rodnet_configs['win_size']
    batch_size = rodnet_configs['batch_size']
    lr = rodnet_configs['learning_rate']
    stacked_num = rodnet_configs['stacked_num']

    if not args.save_memory:
        crdata_train = CRDataset(os.path.join(args.data_dir, 'data_details'),
                                 os.path.join(args.data_dir, 'confmaps_gt'),
                                 win_size=win_size, set_type='train', stride=8)
        seq_names = crdata_train.seq_names
        index_mapping = crdata_train.index_mapping
        dataloader = DataLoader(crdata_train, batch_size=batch_size, shuffle=True, num_workers=0)
    else:
        crdata_train = CRDatasetSM(os.path.join(args.data_dir, 'data_details'),
                                 os.path.join(args.data_dir, 'confmaps_gt'),
                                 win_size=win_size, set_type='train', stride=8, is_Memory_Limit=True)
        seq_names = crdata_train.seq_names
        index_mapping = crdata_train.index_mapping
        dataloader = CRDataLoader(crdata_train, batch_size=batch_size, shuffle=True)

    # print training configurations
    print("Number of sequences to train: %d" % crdata_train.n_seq)
    print("Training files length: %d" % len(crdata_train))
    print("Window size: %d" % win_size)
    print("Number of epoches: %d" % n_epoch)
    print("Batch size: %d" % batch_size)
    print("Number of iterations in each epoch: %d" % int(len(crdata_train) / batch_size))

    if args.model == 'CDC':
        rodnet = RODNet(n_class=n_class, win_size=win_size).cuda()
        criterion = nn.MSELoss()
        stacked_num = 1
    elif args.model == 'HG':
        rodnet = RODNet(n_class=n_class, win_size=win_size, stacked_num=stacked_num).cuda()
        criterion = nn.BCELoss()
    elif args.model == 'HGwI':
        rodnet = RODNet(n_class=n_class, win_size=win_size, stacked_num=stacked_num).cuda()
        criterion = nn.BCELoss()
    else:
        raise TypeError

    # criterion = FocalLoss(focusing_param=8, balance_param=0.25)
    # criterion = nn.CrossEntropyLoss()
    # criterion2 = nn.SmoothL1Loss()
    optimizer = optim.Adam(rodnet.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=rodnet_configs['lr_step'], gamma=0.1)

    iter_count = 0
    if model_name is not None:
        checkpoint = torch.load(os.path.join(train_model_path, '%s/%s' % (model_dir, model_name)))
        if 'optimizer_state_dict' in checkpoint:
            rodnet.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_start = checkpoint['epoch']
            iter_start = checkpoint['iter']
            loss_cp = checkpoint['loss']
            if 'iter_count' in checkpoint:
                iter_count = checkpoint['iter_count']
        else:
            rodnet.load_state_dict(checkpoint)

    for epoch in range(epoch_start, n_epoch):

        tic_load = time.time()
        # if epoch == epoch_start:
        #     dataloader_start = iter_start
        # else:
        #     dataloader_start = 0

        for iter, (data, confmap_gt, obj_info, real_id) in enumerate(dataloader):

            flag = False
            for id in real_id:
                if id == -1:
                    # in case load npy fail
                    print("Warning: Loading NPY data failed! Skip this iteration")
                    tic_load = time.time()
                    flag = True
                    break
            if flag:
                continue

            tic = time.time()
            optimizer.zero_grad()  # zero the parameter gradients
            confmap_preds = rodnet(data.float().cuda())

            loss_confmap = 0
            if stacked_num > 1:
                for i in range(stacked_num):
                    loss_cur = criterion(confmap_preds[i], confmap_gt.float().cuda())
                    loss_confmap += loss_cur
            else:
                loss_cur = criterion(confmap_preds, confmap_gt.float().cuda())
                loss_confmap += loss_cur

            loss_confmap.backward()
            optimizer.step()
            writer.add_scalar('data/loss_all', loss_confmap.item(), iter_count)
            writer.add_scalar('data/loss_confmap', loss_cur.item(), iter_count)
            iter_count += 1

            # print statistics
            print('epoch %d, iter %d: loss: %.8f | load time: %.4f | backward time: %.4f' %
                  (epoch + 1, iter + 1, loss_confmap.item(), tic - tic_load, time.time() - tic))

            if (epoch == epoch_start and iter < 100) or iter % 100 == 0:
                if stacked_num > 1:
                    confmap_pred = confmap_preds[stacked_num - 1].cpu().detach().numpy()
                else:
                    confmap_pred = confmap_preds.cpu().detach().numpy()
                chirp_amp_curr = chirp_amp(data.numpy()[0, :, 0, :, :])

                if True:
                    # draw train images
                    fig_name = os.path.join(train_viz_path,
                                            '%03d_%010d_%06d.png' % (epoch+1, iter_count, iter+1))
                    seq_id, data_id, ra_frame_offset = index_mapping[real_id[0]]
                    img_path = os.path.join(train_sets['root_dir'], seq_names[seq_id][:10], seq_names[seq_id],
                                camera_configs['image_folder'], '%010d.jpg' % data_id)
                    try:
                        visualize_train_img(fig_name, img_path, chirp_amp_curr,
                                            confmap_pred[0, :, 0, :, :], confmap_gt[0, :, 0, :, :])
                    except:
                        img_path = os.path.join(train_sets['root_dir'], seq_names[seq_id][:10], seq_names[seq_id],
                                'images_hist_0', '%010d.jpg' % (data_id + 40))
                        visualize_train_img(fig_name, img_path, chirp_amp_curr,
                                            confmap_pred[0, :, 0, :, :], confmap_gt[0, :, 0, :, :])
                else:
                    writer.add_image('images/ramap', heatmap2rgb(chirp_amp_curr), iter_count)
                    writer.add_image('images/confmap_pred', prob2image(confmap_pred[0, :, 0, :, :]), iter_count)
                    writer.add_image('images/confmap_gt', prob2image(confmap_gt[0, :, 0, :, :]), iter_count)

                    # TODO: combine three images together
                    # writer.add_images('')

            if iter % 1000 == 0:
                status_dict = {
                    'epoch': epoch,
                    'iter': iter,
                    'model_state_dict': rodnet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_confmap,
                    'iter_count': iter_count,
                    }
                save_model_path = os.path.join(train_model_path,
                                               '%s/rodnet_%02d_%010d_%06d.pkl' %
                                               (model_dir, epoch+1, iter_count, iter+1))
                torch.save(status_dict, save_model_path)

            tic_load = time.time()

        scheduler.step()

    print('Training Finished.')

