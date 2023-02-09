import scipy.io as sio
import os
import numpy as np
import cv2
import torch
from torch.utils import data
import torchvision.transforms as tf
from PIL import Image
import json
import h5py

Tensor_to_Image = tf.Compose([
    tf.Normalize([0.0, 0.0, 0.0], [1.0/0.229, 1.0/0.224, 1.0/0.225]),
    tf.Normalize([-0.485, -0.456, -0.406], [1.0, 1.0, 1.0]),
    tf.ToPILImage()
])

def tensor_to_image(image):
    image = Tensor_to_Image(image)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

class nyudv2_DepthDataset(data.Dataset):
    def __init__(self, subset='train', transform=None, root_dir=None, predict_center=False, line_type=''):

        self.transform = transform
        self.predict_center = predict_center
        self.root_dir = root_dir

        dataPath = root_dir
        split = sio.loadmat(os.path.join(dataPath, 'splits.mat'))
        if subset == 'train':
            indices = split['trainNdxs'].reshape(-1) - 1
        else:
            indices = split['testNdxs'].reshape(-1) - 1

        print('loading nyu_depth_v2_labeled.mat...')
        data = h5py.File(os.path.join(dataPath, 'nyu_depth_v2_labeled.mat'))
        print('loading images...')
        images = np.array(data['images'])
        print('loading depths...')
        depths = np.array(data['depths'])
        print('loading nyu_depth_v2_labeled.mat successfully...')

        self.images = images
        self.depths = depths

        full_len = self.images.shape[0]
        img_idxs = np.array(range(full_len)).reshape(-1)
        self.img_selected_idxs = img_idxs[indices]

        self.images = self.images[indices]
        self.depths = self.depths[indices]
        self.h = 192
        self.w = 256

        self.precompute_K_inv_dot_xy_1(h=self.h, w=self.w)

    def precompute_K_inv_dot_xy_1(self, h=192, w=256):
        focal_length = 5.8262448167737955e+02
        offset_x = 3.1304475870804731e+02
        offset_y = 2.3844389626620386e+02

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

        # return lines_pad, line_num
        if line_num == 0:
            line_num = 1
        # print('line pt = ', )
        return lines_pad, line_num

    def __getitem__(self, index):
        image = self.images[index].transpose((2, 1, 0)).astype(np.uint8)[:, :, ::1]

        _, h0, w0 = image.shape

        scale_h = 2.5
        scale_w = 2.5

        image = cv2.resize(image, (self.w, self.h))
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)

        depth = self.depths[index].transpose((1, 0)).astype(np.float32)
        depth = cv2.resize(depth, (self.w, self.h))

        img_idx = self.img_selected_idxs[index]
        lines_file_path = os.path.join(self.root_dir, 'line_info', '%d.txt'%(img_idx))
        lines, num_lines = self.get_lines(lines_file_path)  # 200, 4
        lines[:, 0] = lines[:, 0] / scale_w
        lines[:, 1] = lines[:, 1] / scale_h
        lines[:, 2] = lines[:, 2] / scale_w
        lines[:, 3] = lines[:, 3] / scale_h


        sample = {
            'image': image,
            'depth': torch.FloatTensor(depth),
            'num_lines': num_lines,
            'lines': torch.FloatTensor(lines),
            'img_idx': img_idx
        }

        return sample

    def __len__(self):
        return len(self.images)

    def check_lines(self):
        for i in range(len(self.data_list)):
            if self.subset == 'train':
                data_path = self.data_list[i]
            else:
                data_path = str(i) + '.npz'
            data_path = os.path.join(self.root_dir, data_path)

            lines_file_path = data_path.replace(self.subset, self.subset + '_img')
            lines_file_path = lines_file_path.replace('npz', 'txt')

            lines = np.loadtxt(lines_file_path, dtype=np.float32, delimiter=',')
            lines = lines.reshape(-1, 4)
            line_num = lines.shape[0]
            if line_num <= 1:
                lineinfo = np.zeros([1, 4], dtype=np.float32)
                print(lines_file_path)
                # np.savetxt(lines_file_path, lineinfo, fmt='%.3f', delimiter=',')


class nyudv2_DepthDataset_finetune(data.Dataset):
    def __init__(self, subset='train', transform=None, root_dir=None, predict_center=False):
        assert subset in ['train', 'val']

        if subset == 'val':
            subset = 'test'

        self.subset = subset
        self.transform = transform
        self.predict_center = predict_center

        self.dir_anno = os.path.join(root_dir, subset + '_annotations.json')
        with open(self.dir_anno, 'r') as load_f:
            self.anno = json.load(load_f)
        self.dir_rgb_depth = os.path.join(root_dir, self.anno[0]['dir_AB'])
        data_rgb_depth = sio.loadmat(self.dir_rgb_depth)
        self.data_rgbs = data_rgb_depth['rgbs']
        self.data_depths = data_rgb_depth['depths']
        self.name_rgbs = [self.anno[i]['rgb_path'] for i in range(len(self.anno))]

        self.plane_dir = os.path.join(root_dir, 'data_plane')

        self.precompute_K_inv_dot_xy_1()

        # load gt depth map
        dataPath = '/data/users/tanbin/dataset/NYU_official'
        split = sio.loadmat(dataPath + '/splits.mat')
        if subset == 'train':
            indices = split['trainNdxs'].reshape(-1) - 1
        else:
            indices = split['testNdxs'].reshape(-1) - 1
        print('loading nyu_depth_v2_labeled.mat...')
        data = h5py.File(dataPath + '/nyu_depth_v2_labeled.mat')
        # self.images = np.array(data['images'])
        self.depths = np.array(data['depths'])
        print('loading nyu_depth_v2_labeled.mat successfully...')

        full_len = self.depths.shape[0]
        img_idxs = np.array(range(full_len)).reshape(-1)
        self.img_selected_idxs = img_idxs[indices]

        # self.images = self.images[indices]
        self.depths = self.depths[indices]

    def get_plane_parameters(self, plane, plane_nums, segmentation):
        valid_region = segmentation != 20

        plane = plane[:plane_nums]

        # print('*********************************************8')
        # print('1.--------------', plane[0, :])
        # tmp = plane[:, 1].copy()
        # plane[:, 1] = -plane[:, 2]
        # plane[:, 2] = tmp
        # print('2.--------------', plane[0, :])

        # convert plane from n * d to n / d
        # plane_d = np.linalg.norm(plane, axis=1)
        # print('3.--------------', plane_d[0])
        # normalize
        # plane /= plane_d.reshape(-1, 1)
        # print('4.--------------', plane[0, :])
        # n / d
        # plane /= plane_d.reshape(-1, 1)
        # print('5.--------------', plane[0, :])

        h, w = segmentation.shape
        # plane_parameters = np.ones((3, h, w))
        # for i in range(h):
        #     for j in range(w):
        #         d = segmentation[i, j]
        #         if d >= 20: continue
        #         plane_parameters[:, i, j] = plane[d, :]

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
        focal_length_x = 5.1885790117450188e+02
        focal_length_y = 5.1946961112127485e+02
        offset_x = 3.2558244941119034e+02 - 40
        offset_y = 2.5373616633400465e+02 - 44

        K = [[focal_length_x, 0, offset_x],
             [0, focal_length_y, offset_y],
             [0, 0, 1]]

        K_inv = np.linalg.inv(np.array(K))
        self.K_inv = K_inv

        K_inv_dot_xy_1 = np.zeros((3, h, w))
        xy_map = np.zeros((2, h, w))
        for y in range(h):
            for x in range(w):
                yy = float(y) / h * 427 #480
                xx = float(x) / w * 561 #640

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

        # return lines_pad, line_num
        if line_num == 0:
            line_num = 1
        # print('line pt = ', )
        return lines_pad, line_num

    def __getitem__(self, index):
        if self.subset == 'train':
            image_idx = int(self.name_rgbs[index].split('_')[2]) + 1
        else:
            image_idx = int(self.name_rgbs[index].split('_')[1]) + 1
        plane_data_path = os.path.join(self.plane_dir, 'plane_instance_' + str(image_idx) + '.npz')

        image_seg_map_path = os.path.join(self.plane_dir, 'plane_instance_' + str(image_idx) + '.png')
        image_seg_map = cv2.imread(image_seg_map_path)
        image = image_seg_map[:, :256, :]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)

        # gt_depth = np.zeros((192, 256), dtype=np.float32)
        gt_depth = self.depths[index].transpose((1, 0)).astype(np.float32)
        gt_depth = gt_depth[44:471, 40:601]
        gt_depth = cv2.resize(gt_depth, (256, 192))

        data = np.load(plane_data_path)

        plane = data['plane_param']
        num_planes = plane.shape[0]

        gt_segmentation = data['plane_instance']
        gt_segmentation = gt_segmentation - 1
        seg_mak = (gt_segmentation >= 0).astype(np.float32)
        gt_segmentation = gt_segmentation * seg_mak + (1-seg_mak) * 20

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
        gt_depth = gt_depth.reshape(192, 256)
        depth = gt_depth.reshape(1, 192, 256)

        depth_plane = self.plane2depth(plane_parameters, num_planes, gt_segmentation, gt_depth).reshape(1, 192, 256)

        # get line segments
        lines_file_path = plane_data_path.replace('data_plane', 'data_lines')
        lines_file_path = lines_file_path.replace('npz', 'txt')
        lines, num_lines = self.get_lines(lines_file_path)  # 200, 4

        sample = {
            'image': image,
            'num_planes': num_planes,
            'instance': torch.ByteTensor(segmentation),
            # one for planar and zero for non-planar
            'semantic': 1 - torch.FloatTensor(segmentation[num_planes, :, :]).unsqueeze(0),
            'gt_seg': torch.LongTensor(gt_segmentation), # single channel
            'depth': torch.FloatTensor(depth),
            'depth_plane': torch.FloatTensor(depth_plane),
            'plane_parameters': torch.FloatTensor(plane_parameters),
            'valid_region': torch.ByteTensor(valid_region.astype(np.uint8)).unsqueeze(0),
            'plane_instance_parameter': torch.FloatTensor(plane_instance_parameter),
            'gt_plane_instance_centers': torch.FloatTensor(gt_plane_instance_centers),
            'gt_plane_pixel_centers': torch.FloatTensor(gt_plane_pixel_centers),
            'image_name_idx': image_idx,
            'num_lines': num_lines,
            'lines': torch.FloatTensor(lines),
            'data_path': plane_data_path
        }

        return sample

    def __len__(self):
        return len(self.anno)

    def check_lines(self):
        for i in range(len(self.data_list)):
            if self.subset == 'train':
                data_path = self.data_list[i]
            else:
                data_path = str(i) + '.npz'
            data_path = os.path.join(self.root_dir, data_path)

            lines_file_path = data_path.replace(self.subset, self.subset + '_img')
            lines_file_path = lines_file_path.replace('npz', 'txt')

            lines = np.loadtxt(lines_file_path, dtype=np.float32, delimiter=',')
            lines = lines.reshape(-1, 4)
            line_num = lines.shape[0]
            if line_num <= 1:
                lineinfo = np.zeros([1, 4], dtype=np.float32)
                print(lines_file_path)
                # np.savetxt(lines_file_path, lineinfo, fmt='%.3f', delimiter=',')


if __name__ == '__main__':
    transforms = tf.Compose([
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    root_dir = 'dataset/NYU_official/'

    loader = data.DataLoader(
            nyudv2_DepthDataset(subset='train', transform=transforms, root_dir=root_dir, predict_center=False),
            batch_size=1, shuffle=False, num_workers=1, pin_memory=True
        )
