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
