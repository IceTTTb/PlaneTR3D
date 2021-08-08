import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import os
import cv2
import torchvision.transforms as transforms
import torch

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = b
        cmap[i, 1] = g
        cmap[i, 2] = r
    return cmap

Tensor_to_Image = transforms.Compose([
    transforms.Normalize([0.0, 0.0, 0.0], [1.0/0.229, 1.0/0.224, 1.0/0.225]),
    transforms.Normalize([-0.485, -0.456, -0.406], [1.0, 1.0, 1.0]),
    transforms.ToPILImage()
])


def tensor_to_image(image):
    image = Tensor_to_Image(image)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def plot_depth_recall_curve(method_recalls, type='', save_path=None, method_color=None):
    assert type in ['pixel', 'PIXEL', 'Pixel', 'plane', 'PLANE', 'Plane']
    depth_threshold = np.arange(0, 0.65, 0.05)
    title = 'Per-'+type+' Recall(%)'

    pre_defined_recalls = {}
    if type in ['pixel', 'PIXEL', 'Pixel']:
        recall_planeAE = np.array(
            [0., 30.59, 51.88, 62.83, 68.54, 72.13, 74.28, 75.38, 76.57, 77.08, 77.35, 77.54, 77.86])
        pre_defined_recalls['PlaneAE'] = recall_planeAE

        recall_planeNet = np.array(
            [0., 22.79, 42.19, 52.71, 58.92, 62.29, 64.31, 65.20, 66.10, 66.71, 66.96, 67.11, 67.14])
        pre_defined_recalls['PlaneNet'] = recall_planeNet

    else:
        recall_planeAE = np.array(
            [0., 22.93, 40.17, 49.40, 54.58, 57.75, 59.72, 60.92, 61.84, 62.23, 62.56, 62.76, 62.93])
        pre_defined_recalls['PlaneAE'] = recall_planeAE

        recall_planeNet = np.array(
            [0., 15.78, 29.15, 37.48, 42.34, 45.09, 46.91, 47.77, 48.54, 49.02, 49.33, 49.53, 49.59])
        pre_defined_recalls['PlaneNet'] = recall_planeNet

    plt.figure(figsize=(5, 4))
    plt.xlabel('Depth Threshold', fontsize=18)
    plt.ylabel(title, fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    markers = {'PlaneNet': 'o', 'PlaneAE': '*'}
    colors = {'PlaneNet': 'gray', 'PlaneAE': '#FFCC99'}
    for method_name, recalls in pre_defined_recalls.items():
        assert len(depth_threshold) == len(recalls)
        plt.plot(depth_threshold, recalls, linewidth=3, marker=markers[method_name],label=method_name, color=colors[method_name])

    for method_name, recalls in method_recalls.items():
        assert len(depth_threshold) == len(recalls)
        if method_color is not None:
            plt.plot(depth_threshold, recalls, linewidth=3, marker='^', color=method_color[method_name], label=method_name)
        else:
            plt.plot(depth_threshold, recalls, linewidth=3, marker='^', label=method_name, color='#FF6666')

    plt.legend(loc='lower right', fontsize=16)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    x_major_locator = MultipleLocator(0.1)
    y_major_locator = MultipleLocator(20)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'depth_recall_%s.png'%(type)))
    else:
        plt.savefig('../results/depth_recall_%s.png'%(type))
    plt.close()


def plot_normal_recall_curve(method_recalls, type='', save_path=None, method_color=None):
    assert type in ['pixel', 'PIXEL', 'Pixel', 'plane', 'PLANE', 'Plane']
    normal_threshold = np.linspace(0.0, 30, 13)
    title = 'Per-'+type+' Recall(%)'

    pre_defined_recalls = {}
    if type in ['pixel', 'PIXEL', 'Pixel']:
        recall_planeAE = np.array(
            [0., 30.20, 59.89, 69.79, 73.59, 75.67, 76.8, 77.3, 77.42, 77.57, 77.76, 77.85, 78.03])
        pre_defined_recalls['PlaneAE'] = recall_planeAE

        recall_planeNet = np.array(
            [0., 19.68, 43.78, 57.55, 63.36, 65.27, 66.03, 66.64, 66.99, 67.16, 67.20, 67.26, 67.29])
        pre_defined_recalls['PlaneNet'] = recall_planeNet
    else:
        recall_planeAE = np.array(
            [0., 20.05, 42.66, 51.85, 55.92, 58.34, 59.52, 60.35, 60.75, 61.23, 61.64, 61.84, 61.93])
        pre_defined_recalls['PlaneAE'] = recall_planeAE

        recall_planeNet = np.array(
            [0., 12.49, 29.70, 40.21, 44.92, 46.77, 47.71, 48.44, 48.83, 49.09, 49.20, 49.31, 49.38])
        pre_defined_recalls['PlaneNet'] = recall_planeNet

    plt.figure(figsize=(5, 4))
    plt.xlabel('Normal Threshold', fontsize=18)
    plt.ylabel(title, fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    markers = {'PlaneNet': 'o', 'PlaneAE': '*', 'PlaneRCNN': '.'}
    colors = {'PlaneNet': 'gray', 'PlaneAE': '#FFCC99', 'PlaneRCNN': 'mediumaquamarine'}
    for method_name, recalls in pre_defined_recalls.items():
        assert len(normal_threshold) == len(recalls)
        plt.plot(normal_threshold, recalls, linewidth=3, marker=markers[method_name], label=method_name,
                 color=colors[method_name])

    for method_name, recalls in method_recalls.items():
        assert len(normal_threshold) == len(recalls)
        if method_color is not None:
            plt.plot(normal_threshold, recalls, linewidth=3, marker='^', color=method_color[method_name], label=method_name)
        else:
            plt.plot(normal_threshold, recalls, linewidth=3, marker='^', label=method_name, color='#FF6666')

    plt.legend(loc='lower right', fontsize=16)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    x_major_locator = MultipleLocator(5)
    y_major_locator = MultipleLocator(20)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'normal_recall_%s.png'%(type)))
    else:
        plt.savefig('../results//normal_recall_%s.png'%(type))
    plt.close()



def visualizationBatch(root_path, idx, info, data_dict, non_plane_idx,
                       save_image=False, save_segmentation=False, save_depth=False, save_ply=False, save_cloud=False):
    # if not os.path.exists(root_path):
    #     os.makedirs(root_path)

    assert 'image' in data_dict.keys()
    assert non_plane_idx == -1

    image = data_dict['image'].clone()
    assert image.dim() == 4 or image.dim() == 3
    if image.dim() == 4:
        image = tensor_to_image(image.cpu()[0])
    else:
        image = tensor_to_image(image.unsqueeze(0).cpu()[0])

    if save_image:
        img_path = os.path.join(root_path, '%d_image.png'%(idx))
        cv2.imwrite(img_path, image.astype(np.uint8))

    if save_segmentation:
        print("Notice: please ensure that the non-plane idx is -1!")
        assert 'segmentation' in data_dict.keys()
        segmentation = data_dict['segmentation'].clone()  # -1 indicates non-plane region
        assert segmentation.dim() == 3 or segmentation.dim() == 2
        if segmentation.dim() == 3:
            assert segmentation.shape[0] == 1
            segmentation = segmentation[0].cpu().numpy()
        else:
            segmentation = segmentation.cpu().numpy()
        # print(np.unique(segmentation))
        segmentation += 1
        colors = labelcolormap(256)
        # ***************  get color segmentation
        seg = np.stack([colors[segmentation, 0], colors[segmentation, 1], colors[segmentation, 2]], axis=2)
        # ***************  get blend image
        blend_seg = (seg * 0.7 + image * 0.3).astype(np.uint8)
        seg_mask = (segmentation > 0).astype(np.uint8)
        seg_mask = seg_mask[:, :, np.newaxis]
        blend_seg = blend_seg * seg_mask + image.astype(np.uint8) * (1 - seg_mask)
        # ***************  save
        # seg_path = os.path.join(root_path, '%d_seg_%s.png' % (idx, info))
        # cv2.imwrite(seg_path, seg)
        blend_seg_path = os.path.join(root_path, '%d_seg_%s_blend.png' % (idx, info))
        cv2.imwrite(blend_seg_path, blend_seg)

    if save_depth:
        for key in data_dict.keys():
            if 'depth' in key:
                depth = data_dict[key].clone()
                assert depth.dim() == 3 or depth.dim() == 2
                if depth.dim() == 3:
                    depth = depth[0].cpu().numpy()
                else:
                    depth = depth.cpu().numpy()
                depth_color = drawDepthImage(depth)
                depth_mask = depth > 1e-4
                depth_mask = depth_mask[:, :, np.newaxis]
                depth_color = depth_color * depth_mask
                depth_path = os.path.join(root_path, '%d_%s_%s.png' % (idx, key, info))
                cv2.imwrite(depth_path, depth_color)

    if save_ply:
        if not save_depth:
            assert 'depth' in data_dict.keys()
            depth = data_dict['depth'].clone()
            assert depth.dim() == 3 or depth.dim() == 2
            if depth.dim() == 3:
                depth = depth[0].cpu().numpy()
            else:
                depth = depth.cpu().numpy()
        if not save_segmentation:
            assert 'segmentation' in data_dict.keys()
            segmentation = data_dict['segmentation'].clone()  # -1 indicates non-plane region
            assert segmentation.dim() == 3 or segmentation.dim() == 2
            if segmentation.dim() == 3:
                assert segmentation.shape[0] == 1
                segmentation = segmentation[0].cpu().numpy()
            else:
                segmentation = segmentation.cpu().numpy()
            segmentation += 1


        if 'camera' in data_dict.keys():
            cam = data_dict['camera'].clone()
            assert cam.dim() == 2
            assert cam.shape[-1] == 6
            cam = cam[0].cpu().numpy()
        else:
            cam = None

        if 'K_inv_dot_xy_1' in data_dict.keys():
            K_inv_dot_xy_1 = data_dict['K_inv_dot_xy_1'].clone()
            K_inv_dot_xy_1 = K_inv_dot_xy_1.detach().cpu().numpy()
            h, w = depth.shape[-2], depth.shape[-1]
            K_inv_dot_xy_1 = K_inv_dot_xy_1.reshape(3, h, w)

        else:
            K_inv_dot_xy_1 = None

        writePLYFile(root_path, idx, info, depth, segmentation, image, 0, cam, K_inv_dot_xy_1)

    if save_cloud:
        for key in data_dict.keys():
            info_ = info
            if 'depth' in key:
                if not save_depth:
                    depth = data_dict[key].clone()
                    assert depth.dim() == 3 or depth.dim() == 2
                    if depth.dim() == 3:
                        depth = depth[0].cpu().numpy()
                    else:
                        depth = depth.cpu().numpy()

                info_ = key + '_' + info_

                if not save_segmentation:
                    if 'segmentation' in data_dict.keys():
                        # assert 'segmentation' in data_dict.keys()
                        segmentation = data_dict['segmentation'].clone()  # -1 indicates non-plane region
                        assert segmentation.dim() == 3 or segmentation.dim() == 2
                        if segmentation.dim() == 3:
                            assert segmentation.shape[0] == 1
                            segmentation = segmentation[0].cpu().numpy()
                        else:
                            segmentation = segmentation.cpu().numpy()
                        segmentation += 1
                    else:
                        segmentation = np.ones_like(depth)

                if 'camera' in data_dict.keys():
                    cam = data_dict['camera'].clone()
                    assert cam.dim() == 2
                    assert cam.shape[-1] == 6
                    cam = cam[0].cpu().numpy()
                else:
                    cam = None

                if 'K_inv_dot_xy_1' in data_dict.keys():
                    K_inv_dot_xy_1 = data_dict['K_inv_dot_xy_1'].clone()
                    K_inv_dot_xy_1 = K_inv_dot_xy_1.detach().cpu().numpy()
                    h, w = depth.shape[-2], depth.shape[-1]
                    K_inv_dot_xy_1 = K_inv_dot_xy_1.reshape(3, h, w)

                else:
                    K_inv_dot_xy_1 = None

                writePointCloud(root_path, idx, info_, depth, image, segmentation, 0, K_inv_dot_xy_1=K_inv_dot_xy_1,
                                cam=cam)


def drawDepthImage(depth, maxDepth=5):
    depthImage = np.clip(depth / maxDepth * 255, 0, 255).astype(np.uint8)
    depthImage = cv2.applyColorMap(255 - depthImage, colormap=cv2.COLORMAP_JET)
    return depthImage


def get_K_inv_dot_xy1(cam_ori, out_h, out_w):
    out_h = int(out_h)
    out_w = int(out_w)

    fx = cam_ori[0]
    fy = cam_ori[1]
    offset_x = cam_ori[2]
    offset_y = cam_ori[3]
    ori_w = cam_ori[4]
    ori_h = cam_ori[5]

    K = [[fx, 0, offset_x],
         [0, fy, offset_y],
         [0, 0, 1]]

    K_inv = np.linalg.inv(np.array(K))

    K_inv_dot_xy_1 = np.zeros((3, out_h, out_w), dtype=np.float32)

    for y in range(out_h):
        for x in range(out_w):
            yy = float(y) / out_h * ori_h
            xx = float(x) / out_w * ori_w

            ray = np.dot(K_inv, np.array([xx, yy, 1]).reshape(3, 1))
            K_inv_dot_xy_1[:, y, x] = ray[:, 0]

    return K_inv_dot_xy_1


def writePLYFile(folder, imgIdx, type, depth, segmentation, image, nonplane_idx, cam=None, K_inv_dot_xy_1=None):
    assert cam is not None or K_inv_dot_xy_1 is not None

    # out_h, out_w = depth.shape
    out_h = 192
    out_w = 256

    if K_inv_dot_xy_1 is None and cam is not None:
        K_inv_dot_xy_1 = get_K_inv_dot_xy1(cam, out_h, out_w)

    depth = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    segmentation = cv2.resize(segmentation, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    # create face from segmentation
    faces = []
    for y in range(out_h-1):
        for x in range(out_w-1):
            segmentIndex = segmentation[y, x]
            # ignore non planar region
            if segmentIndex == 0:
                continue

            # add face if three pixel has same segmentatioin
            depths = [depth[y][x], depth[y + 1][x], depth[y + 1][x + 1]]
            if segmentation[y + 1, x] == segmentIndex and segmentation[y + 1, x + 1] == segmentIndex and min(depths) > 0 and max(depths) < 10:
                faces.append((x, y, x, y + 1, x + 1, y + 1))

            depths = [depth[y][x], depth[y][x + 1], depth[y + 1][x + 1]]
            if segmentation[y][x + 1] == segmentIndex and segmentation[y + 1][x + 1] == segmentIndex and min(depths) > 0 and max(depths) < 10:
                faces.append((x, y, x + 1, y + 1, x + 1, y))

    with open(folder + '/' + str(imgIdx) + '_' + type + '_model.ply', 'w') as f:
        header = """ply
format ascii 1.0
comment VCGLIB generated
element vertex """
        header += str(out_h * out_w)
        header += """
property float x
property float y
property float z
property uint8 red 
property uint8 green
property uint8 blue 
element face """
        header += str(len(faces))
        header += """
property list uchar int vertex_indices
property list uchar float texcoord
end_header
"""
        f.write(header)
        for y in range(out_h):
            for x in range(out_w):
                segmentIndex = segmentation[y][x]
                if segmentIndex == nonplane_idx:
                    f.write("0.0 0.0 0.0 0 0 0\n")
                    continue
                ray = K_inv_dot_xy_1[:, y, x]
                X, Y, Z = ray * depth[y, x]
                blue, green, red = image[y, x, 0], image[y, x, 1], image[y, x, 2]
                f.write(str(X) + ' ' + str(Y) + ' ' + str(Z) + ' ' + str(red) + ' ' + str(green) + ' ' + str(blue) + '\n')

        for face in faces:
            f.write('3 ')
            for c in range(3):
                f.write(str(face[c * 2 + 1] * out_w + face[c * 2]) + ' ')
            f.write('6 ')
            for c in range(3):
                f.write(str(float(face[c * 2]) / out_w) + ' ' + str(1 - float(face[c * 2 + 1]) / out_h) + ' ')
            f.write('\n')
        f.close()
    return


def writePointCloud(folder, imgIdx, type, depth, image, segmentation, nonplaneIdx, K_inv_dot_xy_1=None, cam=None, extrinsics=None):
    # out_h, out_w = depth.shape
    out_h = 192
    out_w = 256
    depth = np.clip(depth, a_min=1e-10, a_max=10.)

    if K_inv_dot_xy_1 is None:
        assert cam is not None
        K_inv_dot_xy_1 = get_K_inv_dot_xy1(cam, out_h, out_w)

    depth = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    segmentation = cv2.resize(segmentation, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    mask = segmentation == nonplaneIdx
    mask = (1 - mask).astype(depth.dtype)

    points_corr = K_inv_dot_xy_1 * depth[np.newaxis, :, :] * mask[np.newaxis, :, :]  # 3, h, w

    if extrinsics is not None:
        assert extrinsics.shape[0] == extrinsics.shape[1]  # 4, 4
        assert extrinsics.shape[0] == 4
        points_local = points_corr.reshape(3, -1)
        ones = np.ones_like(points_local[0:1])
        points_local_homo = np.concatenate((points_local, ones), axis=0)  # 4, h*w
        ext_inv = np.linalg.inv(extrinsics)  # 4, 4
        # print(extrinsics)
        # print(ext_inv)
        # exit()
        points_global_homo = np.matmul(ext_inv, points_local_homo)  # 4, h*w
        points_global = points_global_homo[:3, :] / (points_global_homo[3, :] + 1e-10)  # 3, h*w
        points_corr = points_global.reshape(3, out_h, out_w)

    points_corr = points_corr * 100  # m -> cm

    points_rgb = image.transpose(2, 0, 1)[::-1, :, :]
    points = np.concatenate((points_corr, points_rgb), axis=0)
    assert points.shape[0] == 6
    points = points.reshape(6, -1).transpose(1, 0)

    filename = folder + '/' + str(imgIdx) + '_' + type + '_points.txt'
    np.savetxt(filename, points, delimiter=' ', fmt='%d')

