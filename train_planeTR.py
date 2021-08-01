import scipy.io as sio
import os
import cv2
import time
import random
import pickle
import numpy as np
from PIL import Image
import yaml
import sys

import torch
from torch.utils import data
import torch.nn.functional as F
import torchvision.transforms as tf

from utils.utils import Set_Config, Set_Logger, Set_Ckpt_Code_Debug_Dir

from models.planeTR_HRNet import PlaneTR_HRNet as PlaneTR
from models.ScanNetV1_PlaneDataset import scannetv1_PlaneDataset

from utils.misc import AverageMeter, get_optimizer, get_coordinate_map

from models.matcher import HungarianMatcher
from models.detrStyleLoss import SetCriterion

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--mode', default='train', type=str,
                    help='train / eval')
parser.add_argument('--backbone', default='hrnet', type=str,
                    help='only support hrnet now')
parser.add_argument('--cfg_path', default='configs/config_planeTR_train.yaml', type=str,
                    help='full path of the config file')
args = parser.parse_args()

NUM_GPUS = torch.cuda.device_count()

torch.backends.cudnn.benchmark = True


def load_dataset(cfg, args):
    transforms = tf.Compose([
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    assert NUM_GPUS > 0

    if args.mode == 'train':
        subset = 'train'
    else:
        subset = 'val'

    if NUM_GPUS > 1:
        is_shuffle = False
    else:
        is_shuffle = subset == 'train'

    if cfg.dataset.name == 'scannet':
        dataset = scannetv1_PlaneDataset
    else:
        print("undefined dataset!")
        exit()

    predict_center = cfg.model.if_predict_center

    if NUM_GPUS > 1:
        assert args.mode == 'train'
        dataset_plane = dataset(subset=subset, transform=transforms, root_dir=cfg.dataset.root_dir, predict_center=predict_center)
        data_sampler = torch.utils.data.distributed.DistributedSampler(dataset_plane)
        loaders = torch.utils.data.DataLoader(dataset_plane, batch_size=cfg.dataset.batch_size, shuffle=is_shuffle,
                                                   num_workers=cfg.dataset.num_workers, pin_memory=True, sampler=data_sampler)
    else:
        loaders = data.DataLoader(
            dataset(subset=subset, transform=transforms, root_dir=cfg.dataset.root_dir, predict_center=predict_center),
            batch_size=cfg.dataset.batch_size, shuffle=is_shuffle, num_workers=cfg.dataset.num_workers, pin_memory=True
        )
        data_sampler = None

    return loaders, data_sampler


def train(cfg, logger):
    logger.info('*' * 40)
    localtime = time.asctime(time.localtime(time.time()))
    logger.info(localtime)
    logger.info('start training......')
    logger.info('*' * 40)

    model_name = (cfg.save_path).split('/')[-1]

    # set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set ckpt/code/debug dir to save
    checkpoint_dir = Set_Ckpt_Code_Debug_Dir(cfg, args, logger)

    # build network
    network = PlaneTR(cfg)

    # load nets into gpu
    if NUM_GPUS > 1:
        network = DDP(network.to(device), device_ids=[args.local_rank], find_unused_parameters=True)
    else:
        network = network.to(device)

    # load pretrained weights if existed
    if not (cfg.resume_dir == 'None'):
        loc = 'cuda:{}'.format(args.local_rank)
        model_dict = torch.load(cfg.resume_dir, map_location=loc)
        model_dict_ = {}
        if NUM_GPUS > 1:
            for k, v in model_dict.items():
                k_ = 'module.' + k
                model_dict_[k_] = v
            network.load_state_dict(model_dict_)
        else:
            network.load_state_dict(model_dict)

    # set up optimizers
    optimizer = get_optimizer(network.parameters(), cfg.solver)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.solver.lr_step, gamma=cfg.solver.gamma)

    # build data loader
    data_loader, data_sampler = load_dataset(cfg, args)

    # set network state
    if_predict_center = cfg.model.if_predict_center
    use_lines = cfg.model.use_lines
    network.train(not cfg.model.fix_bn)

    k_inv_dot_xy1 = get_coordinate_map(device)

    # set losses and cost matcher
    matcher = HungarianMatcher(cost_class=1., cost_param=1.)
    weight_dict = {'loss_ce': 1, 'loss_param_l1': 1, 'loss_param_cos': 5, 'loss_embedding': 5,
                   'loss_Q': 2, 'loss_center_instance': 1, 'loss_center_pixel': 1, 'loss_depth_pixel': 1}  # run 8
    losses = ['labels', 'param', 'embedding', 'Q']
    if if_predict_center:
        losses.append('center')
    if cfg.model.if_predict_depth:
        losses.append('depth')

    criterion = SetCriterion(num_classes=2, matcher=matcher, weight_dict=weight_dict, eos_coef=1, losses=losses,
                             k_inv_dot_xy1=k_inv_dot_xy1)
    logger.info(f"used losses = {weight_dict}")

    # main loop
    start_epoch = 0
    for epoch in range(start_epoch, cfg.num_epochs):
        if NUM_GPUS > 1:
            data_sampler.set_epoch(epoch)

        # --------------------------------------  time log
        batch_time = AverageMeter()

        # --------------------------------------  loss log
        losses = AverageMeter()
        metric_tracker = {'Classify_instance': ('loss_ce', AverageMeter()),
                          'Pull': ('loss_pull', AverageMeter()),
                          'Push': ('loss_push', AverageMeter()),
                          'PlaneParam_L1': ('loss_param_l1', AverageMeter()),
                          'PlaneParam_Cos': ('loss_param_cos', AverageMeter()),
                          'PlaneParam_Q': ('loss_Q', AverageMeter()),
                          'Center_Pixel': ('loss_center_pixel', AverageMeter()),
                          'Center_Plane': ('loss_center_instance', AverageMeter()),
                          'Depth_pixel': ('loss_depth_pixel', AverageMeter()),
                          'PlaneParam_Angle': ('mean_angle', AverageMeter())}

        tic = time.time()
        for iter, sample in enumerate(data_loader):
            image = sample['image'].to(device)  # b, 3, h, w
            instance = sample['instance'].to(device)
            # semantic = sample['semantic'].to(device)
            gt_depth = sample['depth'].to(device)  # b, 1, h, w
            gt_seg = sample['gt_seg'].to(device)
            # gt_plane_parameters = sample['plane_parameters'].to(device)
            valid_region = sample['valid_region'].to(device)
            gt_plane_instance_parameter = sample['plane_instance_parameter'].to(device)
            gt_plane_instance_centers = sample['gt_plane_instance_centers'].to(device)
            gt_plane_pixel_centers = sample['gt_plane_pixel_centers'].to(device)
            num_planes = sample['num_planes']
            data_path = sample['data_path']
            if use_lines:
                num_lines = sample['num_lines']
                lines = sample['lines'].to(device)  # 200, 4
            else:
                num_lines = None
                lines = None

            # forward pass
            outputs = network(image, lines, num_lines)

            # -------------------------------------- data process
            bs = image.size(0)
            targets = []
            for i in range(bs):
                gt_plane_num = int(num_planes[i])
                tgt = torch.ones([gt_plane_num, 6], dtype=torch.float32, device=device)
                tgt[:, 0] = 1
                tgt[:, 1:4] = gt_plane_instance_parameter[i, :gt_plane_num, :]
                tgt[:, 4:] = gt_plane_instance_centers[i, :gt_plane_num, :]
                tgt = tgt.contiguous()
                targets.append(tgt)

            outputs['gt_instance_map'] = instance
            outputs['gt_depth'] = gt_depth
            outputs['gt_plane_pixel_centers'] = gt_plane_pixel_centers
            outputs['valid_region'] = valid_region
            if 'aux_outputs' in outputs.keys():
                for i, _ in enumerate(outputs['aux_outputs']):
                    outputs['aux_outputs'][i]['gt_instance_map'] = instance

            # calculate losses
            loss_dict, _, loss_dict_aux = criterion(outputs, targets)
            if loss_dict_aux:
                loss_lastLayer = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                loss_aux = 0.
                aux_weight = cfg.aux_weight
                for li in range(len(loss_dict_aux)):
                    loss_aux_li = sum(loss_dict_aux[li][k] * weight_dict[k] for k in loss_dict_aux[li].keys() if k in weight_dict)
                    loss_aux += (loss_aux_li * aux_weight)

                loss_final = loss_lastLayer + loss_aux
            else:
                loss_final = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # --------------------------------------  Backward
            optimizer.zero_grad()
            loss_final.backward()
            optimizer.step()

            # --------------------------------------  update losses and metrics
            losses.update(loss_final.item())

            for name_log in metric_tracker.keys():
                name_loss = metric_tracker[name_log][0]
                if name_loss in loss_dict.keys():
                    loss_cur = float(loss_dict[name_loss])
                    metric_tracker[name_log][1].update(loss_cur)

            # -------------------------------------- update time
            batch_time.update(time.time() - tic)
            tic = time.time()

            # ------------------------------------ log information
            if iter % cfg.print_interval == 0 and args.local_rank == 0:
                # print(data_path)
                log_str = f"[{epoch:2d}][{iter:5d}/{len(data_loader):5d}] " \
                          f"Time: {batch_time.val:.2f} ({batch_time.avg:.2f}) " \
                          f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "

                for name_log, (_, tracker) in metric_tracker.items():
                    log_str += f"{name_log}: {tracker.val:.4f} ({tracker.avg:.4f}) "
                logger.info(log_str)

                print(f"[{model_name}-> {epoch:2d}][{iter:5d}/{len(data_loader):5d}] "
                      f"Time: {batch_time.val:.2f} ({batch_time.avg:.2f}) "
                      f"Loss: {losses.val:.4f} ({losses.avg:.4f}) ")
                logger.info('-------------------------------------')

        lr_scheduler.step()

        # log for one epoch
        logger.info('*' * 40)
        log_str = f"[{epoch:2d}] " \
                  f"Loss: {losses.avg:.4f} "
        for name_log, (_, tracker) in metric_tracker.items():
            log_str += f"{name_log}: {tracker.avg:.4f} "
        logger.info(log_str)
        logger.info('*' * 40)

        # save checkpoint
        if cfg.save_model and args.local_rank == 0:
            if (epoch + 1) % cfg.save_step == 0 or epoch >= 58:
                if NUM_GPUS > 1:
                    torch.save(network.module.state_dict(), os.path.join(checkpoint_dir, f"network_epoch_{epoch}.pt"))
                else:
                    torch.save(network.state_dict(), os.path.join(checkpoint_dir, f"network_epoch_{epoch}.pt"))


if __name__ == '__main__':
    cfg = Set_Config(args)

    # ------------------------------------------- set distribution
    if args.mode == 'train' and NUM_GPUS > 1:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP


        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        print('initialize DDP successfully... ')

    # ------------------------------------------ set logger
    logger = Set_Logger(args, cfg)

    # ------------------------------------------ main
    if args.mode == 'train':
        train(cfg, logger)
    else:
        exit()


