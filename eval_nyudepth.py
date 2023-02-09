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
from models.NYUDV2_depth_dataset import nyudv2_DepthDataset

from utils.metric import evaluateDepths

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--mode', default='eval', type=str,
                    help='train / eval')
parser.add_argument('--backbone', default='hrnet', type=str,
                    help='only support hrnet now')
parser.add_argument('--cfg_path', default='configs/config_nyudepth.yaml', type=str,
                    help='full path of the config file')
args = parser.parse_args()

NUM_GPUS = torch.cuda.device_count()

torch.backends.cudnn.benchmark = True

def get_coordinate_map_NYU(device, h, w):
    focal_length = 5.8262448167737955e+02
    offset_x = 3.1304475870804731e+02
    offset_y = 2.3844389626620386e+02

    # focal_length = 517.97
    # offset_x = 320
    # offset_y = 240

    K = [[focal_length, 0, offset_x],
         [0, focal_length, offset_y],
         [0, 0, 1]]
    K_inv = np.linalg.inv(np.array(K))

    K = torch.FloatTensor(K).to(device)
    K_inv = torch.FloatTensor(K_inv).to(device)

    x = torch.arange(w, dtype=torch.float32).view(1, w) / w * 640
    y = torch.arange(h, dtype=torch.float32).view(h, 1) / h * 480

    x = x.to(device)
    y = y.to(device)
    xx = x.repeat(h, 1)
    yy = y.repeat(1, w)
    xy1 = torch.stack((xx, yy, torch.ones((h, w), dtype=torch.float32).to(device)))  # (3, h, w)
    xy1 = xy1.view(3, -1)  # (3, h*w)

    k_inv_dot_xy1 = torch.matmul(K_inv, xy1)  # (3, h*w)
    return k_inv_dot_xy1

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

    dataset = nyudv2_DepthDataset
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

def eval_depth(cfg, logger):
    logger.info('*' * 40)
    localtime = time.asctime(time.localtime(time.time()))
    logger.info(localtime)
    logger.info('start eval depth on NYUDv2 dataset......')
    logger.info('*' * 40)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build network
    network = PlaneTR(cfg)

    # load nets into gpu
    network.to(device)

    # load pretrained weights if existed
    if not (cfg.resume_dir == 'None'):
        loc = 'cuda:{}'.format(args.local_rank)
        # model_dict = torch.load(cfg.resume_dir, map_location=loc)
        model_dict = torch.load(cfg.resume_dir)
        model_dict_ = {}
        if NUM_GPUS > 1:
            for k, v in model_dict.items():
                k_ = 'module.' + k
                model_dict_[k_] = v
            network.load_state_dict(model_dict_, strict=False)
        else:
            network.load_state_dict(model_dict, strict=False)


    # data loader
    data_loader, _ = load_dataset(cfg, args)

    # set network state
    use_lines = cfg.model.use_lines
    network.eval()

    num_queries = cfg.model.num_queries
    embedding_dist_threshold = cfg.model.embedding_dist_threshold

    h_in = data_loader.dataset.h
    w_in = data_loader.dataset.w

    k_inv_dot_xy1 = get_coordinate_map_NYU(device, h=h_in, w=w_in)

    logger.info('data: %s, data len = %d' % (cfg.dataset.name, len(data_loader)))

    non_plane_img_list = []
    depth_statistics = []

    with torch.no_grad():
        for iter, sample in enumerate(data_loader):
            print("processing image %d"%(iter))
            image = sample['image'].to(device)
            gt_depth = sample['depth'].to(device).unsqueeze(1)
            if use_lines:
                num_lines = sample['num_lines']
                lines = sample['lines'].to(device)  # 200, 4
            else:
                num_lines = None
                lines = None

            bs, _, h, w = image.shape
            assert bs == 1, "batch size should be 1 when testing!"

            img_idx = sample['img_idx']

            outputs = network(image, lines, num_lines)

            # decompose outputs
            pred_logits = outputs['pred_logits'][0]  # num_queries, 3
            pred_param = outputs['pred_param'][0]  # num_queries, 3
            pred_plane_embedding = outputs['pred_plane_embedding'][0]  # num_queries, 2
            pred_pixel_embedding = outputs['pixel_embedding'][0]  # 2, h, w
            c_embedding = pred_plane_embedding.shape[-1]
            assert 'pixel_depth' in outputs.keys()
            pred_pixel_depth = outputs['pixel_depth'][0, 0]  # h, w

            # remove non-plane instance
            pred_prob = F.softmax(pred_logits, dim=-1)  # num_queries, 3
            score, labels = pred_prob.max(dim=-1)
            labels[labels != 1] = 0
            label_mask = labels > 0
            if sum(label_mask) == 0:
                non_plane_img_list.append(iter)
                _, max_pro_idx = pred_prob[:, 1].max(dim=0)
                label_mask[max_pro_idx] = 1
            valid_param = pred_param[label_mask, :]  # valid_plane_num, 3
            valid_plane_embedding = pred_plane_embedding[label_mask, :]  # valid_plane_num, c_embedding
            valid_plane_num = valid_plane_embedding.shape[0]
            valid_plane_prob = score[label_mask]  # valid_plane_num
            assert valid_plane_num <= num_queries

            # calculate dist map
            flat_pixel_embedding = pred_pixel_embedding.view(c_embedding, -1).t()  # hw, c_embedding
            dist_map_pixel2planes = torch.cdist(flat_pixel_embedding, valid_plane_embedding, p=2)  # hw, valid_plane_num
            dist_pixel2onePlane, planeIdx_pixel2onePlane = dist_map_pixel2planes.min(-1)  # [hw,]
            dist_pixel2onePlane = dist_pixel2onePlane.view(h, w)  # h, w
            planeIdx_pixel2onePlane = planeIdx_pixel2onePlane.view(h, w)  # h, w
            mask_pixelOnPlane = dist_pixel2onePlane <= embedding_dist_threshold  # h, w

            # get depth map
            depth_maps = 1. / torch.matmul(valid_param, k_inv_dot_xy1)  # (valid_plane_num, h*w)
            inferred_depth = depth_maps.t()[range(h * w), planeIdx_pixel2onePlane.view(-1)].view(h, w)
            if pred_pixel_depth is not None:
                inferred_depth = inferred_depth * mask_pixelOnPlane.float() + pred_pixel_depth * (1-mask_pixelOnPlane.float())
            else:
                import pdb; pdb.set_trace()
                inferred_depth = inferred_depth * mask_pixelOnPlane.float() + gt_depth[0, 0] * (1-mask_pixelOnPlane.float())

            # get plane segmentation
            gt_depth = gt_depth.cpu().numpy()[0, 0].reshape(h, w)  # h, w
            inferred_depth = inferred_depth.cpu().numpy().reshape(h, w)
            inferred_depth = np.clip(inferred_depth, a_min=1e-4, a_max=10.)

            # depth evaluation
            nyu_mask = torch.zeros((480, 640)).cuda()
            nyu_mask[44:471, 40:601] = 1
            nyu_mask = nyu_mask > 0.5
            nyu_mask = nyu_mask.cpu().numpy()
            gt_depth_resize = cv2.resize(gt_depth, (640, 480))
            inferred_depth_resize = cv2.resize(inferred_depth, (640, 480))
            valid_mask = gt_depth_resize > 1e-4
            valid_depth_mask = valid_mask * nyu_mask

            statistics = evaluateDepths(inferred_depth_resize[valid_depth_mask], gt_depth_resize[valid_depth_mask], False)
            depth_statistics.append(statistics)

        logger.info("========================================")
        depth_res = np.array(depth_statistics).mean(0).tolist()
        res_str = ''
        for i in range(len(depth_res)):
            res_str += '%.3f ' % (depth_res[i])
        print(res_str)
        logger.info("****************************************\n\n")

        print('cfg.resume_dir = ', cfg.resume_dir)

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
    if args.mode == 'eval':
        eval_depth(cfg, logger)
    else:
        exit()


