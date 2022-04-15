import torch
import torch.nn as nn
import sys
import math
import random
sys.path.append('../')
from operator import itemgetter
from config import rodnet_configs, radar_configs, semi_loss_err_reg, err_cor_reg_l1, err_cor_reg_l2, err_cor_reg_l3
from utils.mappings import confmap2ra
from model.loss import _MSE_loss

range_grid = confmap2ra(radar_configs, name='range')
angle_grid = confmap2ra(radar_configs, name='angle')
thrs_rng = 1
thrs_prob = 0.6
thrs_clos = 4
thrs_prob_oft = 0.6
rt_sum = 0.5
dif_upd = 5
small_prob = 0.3
ipt_ser_rng = 8
ipt_ser_agl = 6
loss_type = 'MSE'
MSE_loss = nn.MSELoss()
# MSE_loss = _MSE_loss


def get_gaus_radius(sep_pos, nzero_inds):
    wig_lft = max(sep_pos[3] - 7, 0)
    wig_rgt = min(sep_pos[3] + 7, rodnet_configs['input_asize'] - 1)
    hei_top = max(sep_pos[2] - 5, 0)
    hei_bot = min(sep_pos[2] + 5, rodnet_configs['input_rsize'] - 1)
    obj_wig = torch.nonzero(nzero_inds[sep_pos[0], sep_pos[1], sep_pos[2], wig_lft:wig_rgt + 1])
    obj_hei = torch.nonzero(nzero_inds[sep_pos[0], sep_pos[1], hei_top:hei_bot + 1, sep_pos[3]])

    if len(obj_wig) > 1 and len(obj_hei) > 1:
        radus_wig = int((obj_wig[-1] - obj_wig[0]) / 2)
        radus_hei = int((obj_hei[-1] - obj_hei[0]) / 2)
        # print('work', radus_wig, radus_hei)

        return [int(wig_lft + obj_wig[0]), int(wig_lft + obj_wig[-1]), int(hei_top + obj_hei[0]), \
                int(hei_top + obj_hei[-1]), radus_wig, radus_hei]
    else:
        # print('fail', obj_wig, obj_hei)
        # print(sep_pos)
        # print(range_grid[sep_pos[2]], angle_grid[sep_pos[3]])
        return None


def sort_pos_inds(sep_pos_inds):
    # sep_pos_inds = [batch, class, range, angle]
    # ascending sort according to range index
    rng_sort = sorted(sep_pos_inds, key=itemgetter(2))
    # find all objects at the same range (or very close)
    group_sort = []
    for ic, sep_sort in enumerate(rng_sort):
        if ic == 0:
            new_elem = [sep_sort]
            range_idx = sep_sort[2]
        elif abs(range_grid[sep_sort[2]] - range_grid[range_idx]) < thrs_rng:
            # put them in one list
            new_elem.append(sep_sort)
        else:
            # sort the new_elem in the ascending angle order
            new_elem = sorted(new_elem, key=itemgetter(3))
            group_sort.append(new_elem)
            new_elem = [sep_sort]
            range_idx = sep_sort[2]

    # append the last element
    new_elem = sorted(new_elem, key=itemgetter(3))
    group_sort.append(new_elem)
    return group_sort


def semi_neg_loss_single_frame(input, pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    input_abs = torch.sqrt(torch.pow(input[0, 0, :], 2) + torch.pow(input[0, 1, :], 2))
    pred = torch.clamp(pred, 1.4013e-45, 1)
    pos_inds = gt.eq(1).float()
    nzero_inds = gt.gt(0).float()
    sep_pos_inds = torch.nonzero(pos_inds)
    upd_gt = torch.zeros_like(gt)
    wei_gt = torch.ones_like(gt)
    if len(sep_pos_inds) > 0:
        # decrease the weight of the blind spot of camera
        wei_gt[:, 2, 0:40, 0:15] = small_prob
        wei_gt[:, 2, 0:50, 49:64] = small_prob
        wei_gt[:, :, 108:128, :] = small_prob

        # sort the position of ground truth point according to [angle sort[range sort]]
        grp_pos_inds = sort_pos_inds(sep_pos_inds)
        # TODO: modify to the joint optimization for two ranges
        # for idg, grp_pos in enumerate(grp_pos_inds):
        #     if idg == 0:
        #         rng_cstr = [0, 0]
        #     upd_gt, wei_gt, rng_cstr = search_prediction(grp_pos, nzero_inds, pred, upd_gt, wei_gt, gt, rng_cstr)
        for idg, grp_pos in enumerate(grp_pos_inds):
            if idg == 0:
                rng_cstr = [0, 0]
            if idg % 2 == 0 and idg < len(grp_pos_inds) - 1:
                upd_gt, wei_gt, rng_cstr = semi_joint_search(grp_pos, grp_pos_inds[idg + 1], nzero_inds, pred, upd_gt,
                                                             wei_gt, gt, rng_cstr, input_abs)
            elif idg % 2 == 0:
                upd_gt, wei_gt, rng_cstr = semi_joint_search(grp_pos, None, nzero_inds, pred, upd_gt, wei_gt, gt,
                                                             rng_cstr, input_abs)
            else:
                pass

    if loss_type == 'Focal':
        up_pos_inds = upd_gt.eq(1).float()
        up_neg_inds = upd_gt.lt(1).float()
        up_neg_weights = torch.pow(1 - upd_gt, 4)

        loss = 0
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * up_pos_inds * wei_gt
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * up_neg_weights * up_neg_inds * wei_gt

        num_pos = up_pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    elif loss_type == 'MSE':
        loss = MSE_loss(pred, upd_gt)
    else:
        loss = 0

    return loss


def semi_joint_search(grp_pos1, grp_pos2, nzero_inds, pred, upd_gt, wei_gt, gt, rng_cstr, input):
    upd_gt_pos1, wei_gt_pos1, rng_cstr_pos1, prb_conf_pos1 = search_prediction(grp_pos1, nzero_inds, pred, upd_gt,
                                                                               wei_gt, gt, rng_cstr, input)
    if grp_pos2 is None:
        return upd_gt_pos1, wei_gt_pos1, rng_cstr_pos1
    else:
        upd_gt_pos2, wei_gt_pos2, rng_cstr_pos2, prb_conf_pos2 = search_prediction(grp_pos2, nzero_inds, pred,
                                                                                   upd_gt_pos1, wei_gt_pos1, gt,
                                                                                   rng_cstr_pos1, input)
        # Do the search for pos2 firstly and then restrict the pos1
        new_upd_gt_pos2, new_wei_gt_pos2, new_rng_cstr_pos2, new_prb_conf_pos2 = search_prediction(grp_pos2, nzero_inds,
                                                                                                   pred, upd_gt, wei_gt,
                                                                                                   gt, rng_cstr, input)
        new_upd_gt_pos1, new_wei_gt_pos1, new_rng_cstr_pos1, new_prb_conf_pos1 = search_prediction(grp_pos1, nzero_inds,
                                                                                                   pred, new_upd_gt_pos2,
                                                                                                   new_wei_gt_pos2, gt,
                                                                                                   new_rng_cstr_pos2,
                                                                                                   input, True)
        if new_prb_conf_pos1 + new_prb_conf_pos2 > prb_conf_pos1 + prb_conf_pos2:
            return new_upd_gt_pos1, new_wei_gt_pos1, new_rng_cstr_pos2
        else:
            return upd_gt_pos2, wei_gt_pos2, rng_cstr_pos2


def search_prediction(grp_pos, nzero_inds, pred, upd_gt, wei_gt, gt, rng_cstr, input, IS_reverse=False):
    [pre_up_rng, pre_rng] = rng_cstr
    frt_pos = grp_pos[0]
    result = get_gaus_radius(frt_pos, nzero_inds)

    if result is not None:
        if frt_pos[2] <= semi_loss_err_reg['level1']:
            r_err_top = err_cor_reg_l1['top']
            r_err_bot = err_cor_reg_l1['bot']
        elif frt_pos[2] <= semi_loss_err_reg['level2']:
            r_err_top = err_cor_reg_l2['top']
            r_err_bot = err_cor_reg_l2['bot']
            for sep_pos in grp_pos:
                if sep_pos[1] != 2:
                    r_err_bot = r_err_bot - 10
                    r_err_top = -r_err_top
                    break
            # relative range constraint
            if pre_rng != 0:
                if not IS_reverse:
                    new_r_err_top = pre_up_rng - pre_rng - dif_upd
                    new_r_err_bot = frt_pos[2] - pre_rng + (pre_up_rng - pre_rng)
                    if -r_err_top < new_r_err_top < r_err_bot:
                        r_err_top = -new_r_err_top
                    if -r_err_top < new_r_err_bot < r_err_bot:
                        r_err_bot = new_r_err_bot
                else:
                    new_r_err_bot = pre_up_rng - pre_rng + dif_upd
                    new_r_err_top = (pre_up_rng - pre_rng) - (pre_rng - frt_pos[2])
                    if -r_err_top < new_r_err_top < r_err_bot:
                        r_err_top = -new_r_err_top
                    if -r_err_top < new_r_err_bot < r_err_bot:
                        r_err_bot = new_r_err_bot

        elif frt_pos[2] < semi_loss_err_reg['level3']:
            r_err_top = err_cor_reg_l3['top']
            r_err_bot = err_cor_reg_l3['bot']
            for sep_pos in grp_pos:
                if sep_pos[1] != 2:
                    r_err_bot = r_err_bot - 10
                    r_err_top = -r_err_top
                    break
                    # relative range constraint
            if pre_rng != 0:
                if not IS_reverse:
                    new_r_err_top = pre_up_rng - pre_rng - dif_upd
                    new_r_err_bot = frt_pos[2] - pre_rng + (pre_up_rng - pre_rng)
                    if -r_err_top < new_r_err_top < r_err_bot:
                        r_err_top = -new_r_err_top
                    if -r_err_top < new_r_err_bot < r_err_bot:
                        r_err_bot = new_r_err_bot
                else:
                    new_r_err_bot = pre_up_rng - pre_rng + dif_upd
                    new_r_err_top = (pre_up_rng - pre_rng) - (pre_rng - frt_pos[2])
                    if -r_err_top < new_r_err_top < r_err_bot:
                        r_err_top = -new_r_err_top
                    if -r_err_top < new_r_err_bot < r_err_bot:
                        r_err_bot = new_r_err_bot
        else:
            return upd_gt, wei_gt, rng_cstr, 0

        # get the radius of all objects
        radus_r = []
        radus_a = []
        # print(grp_pos)
        for sep_pos in grp_pos:
            # print('separate indices')
            # print(sep_pos)
            # print(gt[sep_pos[0], sep_pos[1], sep_pos[2], sep_pos[3]])
            sep_Res = get_gaus_radius(sep_pos, nzero_inds)
            if sep_Res is not None:
                [_, _, _, _, radus_a_tp, radus_r_tp] = sep_Res
            else:
                radus_a_tp = 7
                radus_r_tp = 5
                print('no object info')
            radus_r.append(radus_r_tp)
            radus_a.append(radus_a_tp)

        se_sum = []
        for rshift in range(-r_err_top, r_err_bot + 1):
            prb_all = 0
            for ids, sep_pos in enumerate(grp_pos):
                # Update search region
                ser_rng_top = max(sep_pos[2] + rshift - radus_r[ids], 0)
                ser_rng_bot = min(sep_pos[2] + rshift + radus_r[ids], rodnet_configs['input_rsize'] - 1)
                val1 = torch.sum(pred[sep_pos[0], sep_pos[1], ser_rng_top:ser_rng_bot + 1, \
                                 max(sep_pos[3] - 2, 0):min(sep_pos[3] + 3, rodnet_configs['input_asize'])]) \
                       * math.exp(- abs(rshift) / 200.)
                val2 = torch.sum(pred[sep_pos[0], :, ser_rng_top:ser_rng_bot + 1, \
                                 max(sep_pos[3] - 2, 0):min(sep_pos[3] + 3, rodnet_configs['input_asize'])]) \
                       * math.exp(- abs(rshift) / 200.)
                val3 = rt_sum * val1 + (1 - rt_sum) * (val2 - val1)
                prb_all = prb_all + val3
            se_sum.append(prb_all)

        m = torch.argmax(torch.Tensor(se_sum))
        ind_shift = int(-r_err_top + m)
        prb_max = 0.3
        prb_conf = se_sum[m]
        for sep_pos in grp_pos:
            prb_tp = torch.max(pred[sep_pos[0], :, sep_pos[2] + ind_shift, sep_pos[3]])
            if prb_tp > prb_max:
                prb_max = prb_tp

        for ids, sep_pos in enumerate(grp_pos):
            if radus_r[ids] == 0 or radus_a[ids] == 0:
                continue

            # compensate the label position offset
            agl_oft = 0
            rng_oft = 0
            agl_oft_max = 0
            rng_oft_max = 0

            if sep_pos[1] == 2:
                ## search the center of object in the input data
                if sep_pos[2] >= semi_loss_err_reg['level1']:
                    rng_oft_max = 8
                if sep_pos[3] < 54:
                    agl_oft_max = -5
                else:
                    pass

                amp_max = 0
                if agl_oft_max != 0 and rng_oft_max != 0:
                    for roi in range(rng_oft_max+1):
                        for aoi in range(agl_oft_max+1):
                            ipt_ser_rng_min = max(0, sep_pos[2] + ind_shift + roi - ipt_ser_rng)
                            ipt_ser_rng_max = min(rodnet_configs['input_rsize'] - 1, sep_pos[2] + ind_shift + roi + ipt_ser_rng)
                            ipt_ser_agl_min = max(0, sep_pos[3] - aoi - ipt_ser_agl)
                            ipt_ser_agl_max = min(rodnet_configs['input_asize'] - 1, sep_pos[3] - aoi + ipt_ser_agl)
                            tmp_amp = torch.mean(input[ipt_ser_rng_min:ipt_ser_rng_max, ipt_ser_agl_min:ipt_ser_agl_max])
                            if tmp_amp > amp_max:
                                amp_max = tmp_amp
                                rng_oft = roi
                                agl_oft = -aoi
                elif agl_oft_max != 0:
                    roi = 0
                    for aoi in range(agl_oft_max + 1):
                        ipt_ser_rng_min = max(0, sep_pos[2] + ind_shift + roi - ipt_ser_rng)
                        ipt_ser_rng_max = min(rodnet_configs['input_rsize'] - 1,
                                              sep_pos[2] + ind_shift + roi + ipt_ser_rng)
                        ipt_ser_agl_min = max(0, sep_pos[3] - aoi - ipt_ser_agl)
                        ipt_ser_agl_max = min(rodnet_configs['input_asize'] - 1, sep_pos[3] - aoi + ipt_ser_agl)
                        tmp_amp = torch.mean(input[ipt_ser_rng_min:ipt_ser_rng_max, ipt_ser_agl_min:ipt_ser_agl_max])
                        if tmp_amp > amp_max:
                            amp_max = tmp_amp
                            rng_oft = roi
                            agl_oft = -aoi
                elif rng_oft_max != 0:
                    aoi = 0
                    for roi in range(rng_oft_max+1):
                        ipt_ser_rng_min = max(0, sep_pos[2] + ind_shift + roi - ipt_ser_rng)
                        ipt_ser_rng_max = min(rodnet_configs['input_rsize'] - 1,
                                              sep_pos[2] + ind_shift + roi + ipt_ser_rng)
                        ipt_ser_agl_min = max(0, sep_pos[3] - aoi - ipt_ser_agl)
                        ipt_ser_agl_max = min(rodnet_configs['input_asize'] - 1, sep_pos[3] - aoi + ipt_ser_agl)
                        tmp_amp = torch.mean(input[ipt_ser_rng_min:ipt_ser_rng_max, ipt_ser_agl_min:ipt_ser_agl_max])
                        if tmp_amp > amp_max:
                            amp_max = tmp_amp
                            rng_oft = roi
                            agl_oft = -aoi
                else:
                    pass


                # try_prob = random.uniform(0,1)
                # if try_prob < thrs_prob_oft:
                #     if sep_pos[2] >= semi_loss_err_reg['level1']:
                #         rng_oft = 7
                #     if sep_pos[3] < 54:
                #         agl_oft = -5
                #     elif sep_pos[3] > 74:
                #         agl_oft = 5
                #     else:
                #         pass
            # update the ind_shift
            ind_shift = ind_shift + rng_oft
            ina_shift = agl_oft
            # update the gt confidence map
            old_rds_rng_top = min(radus_r[ids], sep_pos[2])
            upd_rds_rng_top = min(radus_r[ids], sep_pos[2] + ind_shift)
            old_rds_rng_bot = min(radus_r[ids], rodnet_configs['input_rsize'] - 1 - sep_pos[2])
            upd_rds_rng_bot = min(radus_r[ids], rodnet_configs['input_rsize'] - 1 - (sep_pos[2] + ind_shift))
            rds_rng_top = min(old_rds_rng_top, upd_rds_rng_top)
            rds_rng_bot = min(old_rds_rng_bot, upd_rds_rng_bot)
            old_rds_agl_lft = min(radus_a[ids], sep_pos[3])
            upd_rds_agl_lft = min(radus_a[ids], sep_pos[3] + ina_shift)
            old_rds_agl_rgt = min(radus_a[ids], rodnet_configs['input_asize'] - 1 - sep_pos[3])
            upd_rds_agl_rgt = min(radus_a[ids], rodnet_configs['input_asize'] - 1 - (sep_pos[3] + ina_shift))
            rds_agl_lft = min(old_rds_agl_lft, upd_rds_agl_lft)
            rds_agl_rgt = min(old_rds_agl_rgt, upd_rds_agl_rgt)

            tmp = torch.max(gt[sep_pos[0], sep_pos[1], sep_pos[2] - rds_rng_top:sep_pos[2] + rds_rng_bot + 1,
                            sep_pos[3] - rds_agl_lft:sep_pos[3] + rds_agl_rgt + 1],
                            upd_gt[sep_pos[0], sep_pos[1],
                            sep_pos[2] + ind_shift - rds_rng_top:sep_pos[2] + ind_shift + rds_rng_bot + 1,
                            sep_pos[3] + ina_shift - rds_agl_lft:sep_pos[3] + ina_shift + rds_agl_rgt + 1])
            # print(gt[sep_pos[0], sep_pos[1], sep_pos[2] - rds_rng_top:sep_pos[2] + rds_rng_bot + 1,
            #                 sep_pos[3] - rds_agl_lft:sep_pos[3] + rds_agl_rgt + 1].shape)
            # print(upd_gt[sep_pos[0], sep_pos[1],
            #                 sep_pos[2] + ind_shift - rds_rng_top:sep_pos[2] + ind_shift + rds_rng_bot + 1,
            #                 sep_pos[3] + ina_shift - rds_agl_lft:sep_pos[3] + ina_shift + rds_agl_rgt + 1].shape)

            upd_gt[sep_pos[0], sep_pos[1],
            sep_pos[2] + ind_shift - rds_rng_top:sep_pos[2] + ind_shift + rds_rng_bot + 1,
            sep_pos[3] + ina_shift - rds_agl_lft:sep_pos[3] + ina_shift + rds_agl_rgt + 1] = tmp

            # update the loss weight for gt confmap
            if prb_max < thrs_prob:
                # weakly supervised the low probability region
                wei_gt[sep_pos[0], sep_pos[1],
                sep_pos[2] + ind_shift - rds_rng_top:sep_pos[2] + ind_shift + rds_rng_bot + 1,
                sep_pos[3] + ina_shift - rds_agl_lft:sep_pos[3] + ina_shift + rds_agl_rgt + 1] = math.sqrt(prb_max)
                # # TODO: return it back code
                # wei_gt[sep_pos[0], sep_pos[1],
                # sep_pos[2] + ind_shift - rds_rng_top:sep_pos[2] + ind_shift + rds_rng_bot + 1, \
                # sep_pos[3] - rds_agl_lft:sep_pos[3] + rds_agl_rgt + 1] = 1
            else:
                wei_gt[sep_pos[0], sep_pos[1],
                sep_pos[2] + ind_shift - rds_rng_top:sep_pos[2] + ind_shift + rds_rng_bot + 1,
                sep_pos[3] + ina_shift - rds_agl_lft:sep_pos[3] + ina_shift + rds_agl_rgt + 1] = 1

        return upd_gt, wei_gt, [sep_pos[2] + ind_shift, sep_pos[2]], prb_conf

    else:
        return upd_gt, wei_gt, rng_cstr, 0


def __neg_loss__(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pred = torch.clamp(pred, 1.4013e-45, 1)
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    # zro_inds = gt.eq(0).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    # zro_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * zro_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    # zro_loss = zro_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        # loss = loss - (pos_loss + neg_loss + zro_loss) / num_pos
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss


def KL_Diverg(pred1, pred2):
    eps = 1e-10
    neg_pred1 = 1 - pred1
    neg_pred2 = 1 - pred2
    p1_div_p2 = torch.div(pred1, pred2) + eps
    neg_p1_div_p2 = torch.div(neg_pred1, neg_pred2) + eps
    KL_d = pred1 * torch.log(p1_div_p2) + neg_pred1 * torch.log(neg_p1_div_p2)

    KL_loss = KL_d.sum()
    # print('KL')
    # print(KL_loss)

    return KL_loss


def __semi_neg_loss__(input, pred, gt):
    ws_loss = 0
    kl_loss = 0
    for i in range(rodnet_configs['win_size']):
        ws_loss = ws_loss + semi_neg_loss_single_frame(input[:, :, i, :, :], pred[:, :, i, :, :], gt[:, :, i, :, :])
        if i % 2 == 0:
            if loss_type == 'Focal':
                kl_loss = kl_loss + KL_Diverg(pred[:,:,i,:,:], pred[:,:,i+1,:,:])
            elif loss_type == 'MSE':
                kl_loss = kl_loss + MSE_loss(pred[:,:,i,:,:], pred[:,:,i+1,:,:])
            else:
                pass

    # loss = ws_loss + min(kl_loss/(rodnet_configs['win_size']/2), ws_loss*0.1)
    loss = ws_loss + kl_loss/(rodnet_configs['win_size']/2)

    return loss


# class FocalLoss(nn.Module):
#     '''nn.Module warpper for focal loss'''
#
#     def __init__(self):
#         super(FocalLoss, self).__init__()
#         self.neg_loss = __neg_loss__
#         self.semi_neg_loss = __semi_neg_loss__
#
#     def forward(self, out, target, label_ord):
#         # loss0 = self.neg_loss(torch.unsqueeze(out[0,:,:,:], 0), torch.unsqueeze(target[0,:,:,:], 0))
#         # loss1 = self.semi_neg_loss(torch.unsqueeze(out[1,:,:,:], 0), torch.unsqueeze(target[1,:,:,:], 0))
#         loss = 0
#         for i in range(rodnet_configs['batch_size']):
#             if int(label_ord[i]) == 0:
#                 loss = loss + self.neg_loss(torch.unsqueeze(out[i, :], 0), torch.unsqueeze(target[i, :], 0))
#             else:
#                 loss = loss + self.semi_neg_loss(torch.unsqueeze(out[i, :], 0), torch.unsqueeze(target[i, :], 0))
#             # print(loss)
#             # input()
#         loss = torch.clamp(loss, 0, 30)
#         return loss
#


class Weak_Supv_Loss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(Weak_Supv_Loss, self).__init__()
        # self.neg_loss = __neg_loss__
        self.semi_neg_loss = __semi_neg_loss__

    def forward(self, input, out, target):
        loss = 0
        for i in range(rodnet_configs['batch_size']):
            loss = loss + self.semi_neg_loss(torch.unsqueeze(input[i, :], 0),
                                             torch.unsqueeze(out[i, :], 0),
                                             torch.unsqueeze(target[i, :], 0))
        # print(loss)
        # loss = torch.clamp(loss, 0, 30)
        return loss


class Cont_Loss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(Cont_Loss, self).__init__()

    def forward(self, out, target):
        loss = 0
        for i in range(out.shape[0]):
            batch_loss = 0
            for j in range(rodnet_configs['win_size']):
                if j % 2 == 0:
                    batch_loss = batch_loss + MSE_loss(out[i, :, j, :, :], target[i, :, j + 1, :, :])
            loss = loss + batch_loss/(rodnet_configs['win_size']/2)
            # loss = loss + batch_loss
        return loss
