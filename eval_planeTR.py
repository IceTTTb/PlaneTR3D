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

from utils.metric import eval_plane_recall_depth, eval_plane_recall_normal, evaluateMasks

from utils.disp import plot_depth_recall_curve, plot_normal_recall_curve


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
parser.add_argument('--cfg_path', default='configs/config_planeTR_eval.yaml', type=str,
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
        # todo: add support for nyu dataset
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


def eval(cfg, logger):
    logger.info('*' * 40)
    localtime = time.asctime(time.localtime(time.time()))
    logger.info(localtime)
    logger.info('start evaluating......')
    logger.info('*' * 40)

    # set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build network
    network = PlaneTR(cfg)

    # load nets into gpu
    network = network.to(device)

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

    # build data loader
    data_loader, _ = load_dataset(cfg, args)

    # set network state
    use_lines = cfg.model.use_lines
    network.eval()

    k_inv_dot_xy1 = get_coordinate_map(device)
    num_queries = cfg.model.num_queries
    embedding_dist_threshold = cfg.model.embedding_dist_threshold

    # define metrics
    pixelDepth_recall_curve = np.zeros((13))
    planeDepth_recall_curve = np.zeros((13, 3))
    pixelNorm_recall_curve = np.zeros((13))
    planeNorm_recall_curve = np.zeros((13, 3))
    plane_Seg_Metric = np.zeros((3))

    with torch.no_grad():
        for iter, sample in enumerate(data_loader):
            print("processing image %d"%(iter))
            image = sample['image'].to(device)
            instance = sample['instance'].to(device)
            gt_seg = sample['gt_seg'].numpy()
            gt_depth = sample['depth'].to(device)
            valid_region = sample['valid_region'].to(device)
            gt_plane_num = sample['num_planes'].int()
            gt_plane_instance_parameter = sample['plane_instance_parameter'].to(device)
            gt_plane_instance_centers = sample['gt_plane_instance_centers'].to(device)[0]  # gt_plane_num, 2
            gt_plane_pixel_centers = sample['gt_plane_pixel_centers'].to(device)[0]  # 2, h, w

            if use_lines:
                num_lines = sample['num_lines']
                lines = sample['lines'].to(device)  # 200, 4
            else:
                num_lines = None
                lines = None

            bs, _, h, w = image.shape
            assert bs == 1, "batch size should be 1 when testing!"
            assert h == 192 and w == 256

            sp_t_s = time.time()

            # forward pass
            outputs = network(image, lines, num_lines)

            # decompose outputs
            pred_logits = outputs['pred_logits'][0]  # num_queries, 3
            pred_param = outputs['pred_param'][0]  # num_queries, 3
            pred_plane_embedding = outputs['pred_plane_embedding'][0]  # num_queries, 2
            pred_pixel_embedding = outputs['pixel_embedding'][0]  # 2, h, w

            assert 'pixel_depth' in outputs.keys()
            pred_pixel_depth = outputs['pixel_depth'][0, 0]  # h, w

            # remove non-plane instance
            pred_prob = F.softmax(pred_logits, dim=-1)  # num_queries, 3
            score, labels = pred_prob.max(dim=-1)
            labels[labels != 1] = 0
            label_mask = labels > 0
            if sum(label_mask) == 0:
                _, max_pro_idx = pred_prob[:, 1].max(dim=0)
                label_mask[max_pro_idx] = 1
            valid_param = pred_param[label_mask, :]  # valid_plane_num, 3
            valid_plane_embedding = pred_plane_embedding[label_mask, :]  # valid_plane_num, c_embedding
            valid_plane_num = valid_plane_embedding.shape[0]
            valid_plane_prob = score[label_mask]  # valid_plane_num
            assert valid_plane_num <= num_queries

            # calculate dist map
            c_embedding = pred_plane_embedding.shape[-1]
            flat_pixel_embedding = pred_pixel_embedding.view(c_embedding, -1).t()  # hw, c_embedding
            dist_map_pixel2planes = torch.cdist(flat_pixel_embedding, valid_plane_embedding, p=2)  # hw, valid_plane_num
            dist_pixel2onePlane, planeIdx_pixel2onePlane = dist_map_pixel2planes.min(-1)  # [hw,]
            dist_pixel2onePlane = dist_pixel2onePlane.view(h, w)  # h, w
            planeIdx_pixel2onePlane = planeIdx_pixel2onePlane.view(h, w)  # h, w
            mask_pixelOnPlane = dist_pixel2onePlane <= embedding_dist_threshold  # h, w

            # get plane segmentation
            gt_seg = gt_seg.reshape(h, w)  # h, w
            predict_segmentation = planeIdx_pixel2onePlane.cpu().numpy()  # h, w
            if int(mask_pixelOnPlane.sum()) < (h * w):  # set plane idx of non-plane pixels as num_queries + 1
                predict_segmentation[mask_pixelOnPlane.cpu().numpy() == 0] = num_queries + 1
            predict_segmentation = predict_segmentation.reshape(h, w)  # h, w (0~num_queries-1:plane idx, num_queries+1:non-plane)

            # get depth map
            depth_maps_inv = torch.matmul(valid_param, k_inv_dot_xy1)
            depth_maps_inv = torch.clamp(depth_maps_inv, min=0.1, max=1e4)
            depth_maps = 1. / depth_maps_inv  # (valid_plane_num, h*w)
            inferred_depth = depth_maps.t()[range(h * w), planeIdx_pixel2onePlane.view(-1)].view(h, w)
            inferred_depth = inferred_depth * mask_pixelOnPlane.float() + pred_pixel_depth * (1-mask_pixelOnPlane.float())

            # get depth maps
            gt_depth = gt_depth.cpu().numpy()[0, 0].reshape(h, w)  # h, w
            inferred_depth = inferred_depth.cpu().numpy().reshape(h, w)
            inferred_depth = np.clip(inferred_depth, a_min=1e-4, a_max=10.)

            # ----------------------------------------------------- evaluation
            # 1 evaluation: plane segmentation
            pixelStatistics, planeStatistics = eval_plane_recall_depth(
                predict_segmentation, gt_seg, inferred_depth, gt_depth, valid_plane_num)
            pixelDepth_recall_curve += np.array(pixelStatistics)
            planeDepth_recall_curve += np.array(planeStatistics)

            # 2 evaluation: plane segmentation
            instance_param = valid_param.cpu().numpy()
            gt_plane_instance_parameter = gt_plane_instance_parameter.cpu().numpy()
            plane_recall, pixel_recall = eval_plane_recall_normal(predict_segmentation, gt_seg,
                                                                            instance_param, gt_plane_instance_parameter,
                                                                            pred_non_plane_idx=num_queries+1)
            pixelNorm_recall_curve += pixel_recall
            planeNorm_recall_curve += plane_recall

            # 3 evaluation: plane segmentation
            plane_Seg_Statistics = evaluateMasks(predict_segmentation, gt_seg, device, pred_non_plane_idx=num_queries+1)
            plane_Seg_Metric += np.array(plane_Seg_Statistics)

            # ------------------------------------ log info
            print(f"RI(+):{plane_Seg_Statistics[0]:.3f} | VI(-):{plane_Seg_Statistics[1]:.3f} | SC(+):{plane_Seg_Statistics[2]:.3f}")


        plane_Seg_Metric = plane_Seg_Metric / len(data_loader)

        logger.info("========================================")
        logger.info(f'cfg.resume_dir = {cfg.resume_dir}')
        logger.info("pixel and plane recall (depth) of all test image")
        logger.info(pixelDepth_recall_curve / len(data_loader))
        logger.info(planeDepth_recall_curve[:, 0] / planeDepth_recall_curve[:, 1])
        logger.info("========================================")
        logger.info("pixel and plane recall (normal) of all test image")
        logger.info(pixelNorm_recall_curve / len(data_loader))
        logger.info(planeNorm_recall_curve[:, 0] / planeNorm_recall_curve[:, 1])
        logger.info("========================================")
        logger.info("plane instance segmentation results")
        logger.info(
            f"RI(+):{plane_Seg_Metric[0]:.3f} | VI(-):{plane_Seg_Metric[1]:.3f} | SC(+):{plane_Seg_Metric[2]:.3f}")
        logger.info("****************************************\n\n")

        if cfg.save_path != 'None':
            if not os.path.exists(cfg.save_path):
                os.makedirs(cfg.save_path)
            mine_recalls_pixel = {"PlaneTR (Ours)": pixelDepth_recall_curve / len(data_loader) * 100}
            mine_recalls_plane = {"PlaneTR (Ours)": planeDepth_recall_curve[:, 0] / planeDepth_recall_curve[:, 1] * 100}
            plot_depth_recall_curve(mine_recalls_pixel, type='pixel', save_path=cfg.save_path)
            plot_depth_recall_curve(mine_recalls_plane, type='plane', save_path=cfg.save_path)

            normal_recalls_pixel = {"planeTR": pixelNorm_recall_curve / len(data_loader) * 100}
            normal_recalls_plane = {"planeTR": planeNorm_recall_curve[:, 0] / planeNorm_recall_curve[:, 1] * 100}
            plot_normal_recall_curve(normal_recalls_pixel, type='pixel', save_path=cfg.save_path)
            plot_normal_recall_curve(normal_recalls_plane, type='plane', save_path=cfg.save_path)


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
        eval(cfg, logger)
    else:
        exit()


