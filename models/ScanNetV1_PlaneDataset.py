import os
from torch.utils import data
import numpy as np
import cv2
from PIL import Image
import torch

# modified from https://github.com/svip-lab/PlanarReconstruction/blob/master/main.py
class scannetv1_PlaneDataset(data.Dataset):
    def __init__(self, subset='train', transform=None, root_dir=None, predict_center=False, line_type=''):
        assert subset in ['train', 'val']

        print("*"*20, line_type)
        self.line_type = line_type
        self.subset = subset
        self.transform = transform
        self.predict_center = predict_center
        print("dataloader: predict_center = ", self.predict_center)
        self.root_dir = os.path.join(root_dir, subset)
        self.txt_file = os.path.join(root_dir, subset + '.txt')

        self.data_list = [line.strip() for line in open(self.txt_file, 'r').readlines()]
        self.precompute_K_inv_dot_xy_1()


    def get_plane_parameters(self, plane, plane_nums, segmentation):
        valid_region = segmentation != 20

        plane = plane[:plane_nums]

        tmp = plane[:, 1].copy()
        plane[:, 1] = -plane[:, 2]
        plane[:, 2] = tmp

        # convert plane from n * d to n / d
        plane_d = np.linalg.norm(plane, axis=1)
        # normalize
        plane /= plane_d.reshape(-1, 1)
        # n / d
        plane /= plane_d.reshape(-1, 1)

        h, w = segmentation.shape

        plane_parameters2 = np.ones((3, h, w))
        for i in range(plane_nums):
            plane_mask = segmentation == i
            plane_mask = plane_mask.astype(np.float32)
            cur_plane_param_map = np.ones((3, h, w)) * plane[i, :].reshape(3, 1, 1)
            plane_parameters2 = plane_parameters2 * (1-plane_mask) + cur_plane_param_map * plane_mask

        # plane_instance parameter, padding zero to fix size
        plane_instance_parameter = np.concatenate((plane, np.zeros((20 - plane.shape[0], 3))), axis=0)
        return plane_parameters2, valid_region, plane_instance_parameter

    def precompute_K_inv_dot_xy_1(self, h=192, w=256):
        focal_length = 517.97
        offset_x = 320
        offset_y = 240

        K = [[focal_length, 0, offset_x],
             [0, focal_length, offset_y],
             [0, 0, 1]]

        K_inv = np.linalg.inv(np.array(K))
        self.K_inv = K_inv

        K_inv_dot_xy_1 = np.zeros((3, h, w))
        xy_map = np.zeros((2, h, w))
        for y in range(h):
            for x in range(w):
                yy = float(y) / h * 480
                xx = float(x) / w * 640

                ray = np.dot(self.K_inv,
                             np.array([xx, yy, 1]).reshape(3, 1))
                K_inv_dot_xy_1[:, y, x] = ray[:, 0]
                xy_map[0, y, x] = float(x) / w
                xy_map[1, y, x] = float(y) / h

        # precompute to speed up processing
        self.K_inv_dot_xy_1 = K_inv_dot_xy_1
        self.xy_map = xy_map

    def plane2depth(self, plane_parameters, num_planes, segmentation, gt_depth, h=192, w=256):

        depth_map = 1. / np.sum(self.K_inv_dot_xy_1.reshape(3, -1) * plane_parameters.reshape(3, -1), axis=0)
        depth_map = depth_map.reshape(h, w)

        # replace non planer region depth using sensor depth map
        depth_map[segmentation == 20] = gt_depth[segmentation == 20]
        return depth_map

    def get_lines(self, line_path):
        lines = np.loadtxt(line_path, dtype=np.float32, delimiter=',')
        lines = lines.reshape(-1, 4)

        lines_pad = np.zeros([200, 4], dtype=np.float32)
        line_num = lines.shape[0]
        if line_num == 0:
            pass
        elif line_num > 200:
            lines_pad = lines[0:200, :]
            line_num = 200
        else:
            lines_pad[0:line_num, :] = lines

        if line_num == 0:
            line_num = 1
        return lines_pad, line_num

    def __getitem__(self, index):
        if self.subset == 'train':
            data_path = self.data_list[index]
        else:
            data_path = self.data_list[index]
        data_path = os.path.join(self.root_dir, data_path)
        data = np.load(data_path)
        image = data['image']
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_ori = image.copy()
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)

        plane = data['plane']
        num_planes = data['num_planes'][0]

        gt_segmentation = data['segmentation']
        gt_segmentation = gt_segmentation.reshape((192, 256))
        segmentation = np.zeros([21, 192, 256], dtype=np.uint8)

        # get segmentation: 21, h ,w
        _, h, w = segmentation.shape
        for i in range(num_planes + 1):
            # deal with backgroud
            if i == num_planes:
                seg = gt_segmentation == 20
            else:
                seg = gt_segmentation == i
            segmentation[i, :, :] = seg.reshape(h, w)

        # get plane center
        gt_plane_instance_centers = np.zeros([21, 2])
        gt_plane_pixel_centers = np.zeros([2, 192, 256], dtype=np.float)  # 2, 192, 256
        if self.predict_center:
            for i in range(num_planes):
                plane_mask = gt_segmentation == i
                pixel_num = plane_mask.sum()
                plane_mask = plane_mask.astype(np.float)
                x_map = self.xy_map[0] * plane_mask
                y_map = self.xy_map[1] * plane_mask
                x_sum = x_map.sum()
                y_sum = y_map.sum()
                plane_x = x_sum / pixel_num
                plane_y = y_sum / pixel_num
                gt_plane_instance_centers[i, 0] = plane_x
                gt_plane_instance_centers[i, 1] = plane_y

                center_map = np.zeros([2, 192, 256], dtype=np.float)  # 2, 192, 256
                center_map[0, :, :] = plane_x
                center_map[1, :, :] = plane_y
                center_map = center_map * plane_mask  # 2, 192, 256

                gt_plane_pixel_centers = gt_plane_pixel_centers * (1 - plane_mask) + center_map

        # surface plane parameters
        plane_parameters, valid_region, plane_instance_parameter = \
            self.get_plane_parameters(plane, num_planes, gt_segmentation)

        # since some depth is missing, we use plane to recover those depth following PlaneNet
        gt_depth = data['depth'].reshape(192, 256)
        depth = self.plane2depth(plane_parameters, num_planes, gt_segmentation, gt_depth).reshape(1, 192, 256)
        raw_depth = gt_depth.reshape(1, 192, 256)

        # get line segments
        lines_file_path = data_path.replace(self.subset, self.subset + '_img' + self.line_type)
        lines_file_path = lines_file_path.replace('npz', 'txt')
        lines, num_lines = self.get_lines(lines_file_path)  # 200, 4

        sample = {
            'image': image,
            'image_ori': torch.from_numpy(image_ori).permute(2, 0, 1).float(),
            'num_planes': num_planes,
            'instance': torch.ByteTensor(segmentation),
            # one for planar and zero for non-planar
            'semantic': 1 - torch.FloatTensor(segmentation[num_planes, :, :]).unsqueeze(0),
            'gt_seg': torch.LongTensor(gt_segmentation),  # single channel
            'depth': torch.FloatTensor(depth),
            'raw_depth': torch.FloatTensor(raw_depth),
            'plane_parameters': torch.FloatTensor(plane_parameters),
            'valid_region': torch.ByteTensor(valid_region.astype(np.uint8)).unsqueeze(0),
            'plane_instance_parameter': torch.FloatTensor(plane_instance_parameter),
            'gt_plane_instance_centers': torch.FloatTensor(gt_plane_instance_centers),
            'gt_plane_pixel_centers': torch.FloatTensor(gt_plane_pixel_centers),
            'num_lines': num_lines,
            'lines': torch.FloatTensor(lines),
            'data_path': data_path
        }

        return sample

    def __len__(self):
        return len(self.data_list)
