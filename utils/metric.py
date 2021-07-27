import numpy as np
import torch

# https://github.com/davisvideochallenge/davis/blob/master/python/lib/davis/measures/jaccard.py
def eval_iou(annotation, segmentation):
    """ Compute region similarity as the Jaccard Index.

    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.

    Return:
        jaccard (float): region similarity

    """
    annotation = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)

    if np.isclose(np.sum(annotation), 0) and np.isclose(np.sum(segmentation), 0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / \
               np.sum((annotation | segmentation), dtype=np.float32)


# https://github.com/svip-lab/PlanarReconstruction/blob/master/utils/metric.py
# https://github.com/art-programmer/PlaneNet/blob/master/utils.py#L2115
def eval_plane_recall_depth(predSegmentations, gtSegmentations, predDepths, gtDepths, pred_plane_num, threshold=0.5):
    predNumPlanes = pred_plane_num  # actually, it is the maximum number of the predicted planes

    if 20 in np.unique(gtSegmentations):  # in GT plane Seg., number '20' indicates non-plane
        gtNumPlanes = len(np.unique(gtSegmentations)) - 1
    else:
        gtNumPlanes = len(np.unique(gtSegmentations))

    if len(gtSegmentations.shape) == 2:
        gtSegmentations = (np.expand_dims(gtSegmentations, -1) == np.arange(gtNumPlanes)).astype(np.float32)  # h, w, gtNumPlanes
    if len(predSegmentations.shape) == 2:
        predSegmentations = (np.expand_dims(predSegmentations, -1) == np.arange(predNumPlanes)).astype(np.float32)  # h, w, predNumPlanes

    planeAreas = gtSegmentations.sum(axis=(0, 1))  # gt plane pixel number

    intersectionMask = np.expand_dims(gtSegmentations, -1) * np.expand_dims(predSegmentations, 2) > 0.5  # h, w, gtNumPlanes, predNumPlanes

    depthDiffs = gtDepths - predDepths  # h, w
    depthDiffs = depthDiffs[:, :, np.newaxis, np.newaxis]  # h, w, 1, 1

    intersection = np.sum((intersectionMask).astype(np.float32), axis=(0, 1))  # gtNumPlanes, predNumPlanes

    planeDiffs = np.abs(depthDiffs * intersectionMask).sum(axis=(0, 1)) / np.maximum(intersection, 1e-4)  # gtNumPlanes, predNumPlanes
    planeDiffs[intersection < 1e-4] = 1

    union = np.sum(
        ((np.expand_dims(gtSegmentations, -1) + np.expand_dims(predSegmentations, 2)) > 0.5).astype(np.float32),
        axis=(0, 1))  # gtNumPlanes, predNumPlanes
    planeIOUs = intersection / np.maximum(union, 1e-4)  # gtNumPlanes, predNumPlanes

    numPredictions = int(predSegmentations.max(axis=(0, 1)).sum())

    numPixels = planeAreas.sum()
    IOUMask = (planeIOUs > threshold).astype(np.float32)

    minDiff = np.min(planeDiffs * IOUMask + 1000000 * (1 - IOUMask), axis=1)

    stride = 0.05
    pixelRecalls = []
    planeStatistics = []
    for step in range(int(0.61 / stride + 1)):
        diff = step * stride
        pixelRecalls.append(np.minimum((intersection * (planeDiffs <= diff).astype(np.float32) * IOUMask).sum(1),
                                       planeAreas).sum() / numPixels)
        planeStatistics.append(((minDiff <= diff).sum(), gtNumPlanes, numPredictions))
    return pixelRecalls, planeStatistics

# https://github.com/svip-lab/PlanarReconstruction/blob/master/utils/metric.py
def eval_plane_recall_normal(segmentation, gt_segmentation, param, gt_param, pred_non_plane_idx, threshold=0.5):
    """
    :param segmentation: label map for plane segmentation [h, w] where 20 indicate non-planar
    :param gt_segmentation: ground truth label for plane segmentation where 20 indicate non-planar
    :param threshold: value for iou
    :return: percentage of correctly predicted ground truth planes correct plane
    """
    depth_threshold_list = np.linspace(0.0, 30, 13)

    # both prediction and ground truth segmentation contains non-planar region which indicated by label 20
    # so we minus one
    pred_plane_idxs = np.unique(segmentation)
    if pred_non_plane_idx in pred_plane_idxs:
        pred_plane_idx_max = pred_plane_idxs[-2]
    else:
        pred_plane_idx_max = pred_plane_idxs[-1]
    plane_num = pred_plane_idx_max + 1


    if 20 in np.unique(gt_segmentation):  # in GT plane Seg., number '20' indicates non-plane
        gt_plane_num = len(np.unique(gt_segmentation)) - 1
    else:
        gt_plane_num = len(np.unique(gt_segmentation))

    # 13: 0:0.05:0.6
    plane_recall = np.zeros((gt_plane_num, len(depth_threshold_list)))
    pixel_recall = np.zeros((gt_plane_num, len(depth_threshold_list)))

    plane_area = 0.0

    gt_param = gt_param.reshape(20, 3)

    # check if plane is correctly predict
    for i in range(gt_plane_num):
        gt_plane = gt_segmentation == i
        plane_area += np.sum(gt_plane)

        for j in range(plane_num):
            pred_plane = segmentation == j
            iou = eval_iou(gt_plane, pred_plane)

            if iou > threshold:
                # mean degree difference over overlap region:
                gt_p = gt_param[i]
                pred_p = param[j]

                n_gt_p = gt_p / np.linalg.norm(gt_p)
                n_pred_p = pred_p / np.linalg.norm(pred_p)

                angle = np.arccos(np.clip(np.dot(n_gt_p, n_pred_p), -1.0, 1.0))
                degree = np.degrees(angle)
                depth_diff = degree

                # compare with threshold difference
                plane_recall[i] = (depth_diff < depth_threshold_list).astype(np.float32)
                pixel_recall[i] = (depth_diff < depth_threshold_list).astype(np.float32) * \
                                  (np.sum(gt_plane * pred_plane))
                break

    pixel_recall = np.sum(pixel_recall, axis=0).reshape(-1) / plane_area

    plane_recall_new = np.zeros((len(depth_threshold_list), 3))
    plane_recall = np.sum(plane_recall, axis=0).reshape(-1, 1)
    plane_recall_new[:, 0:1] = plane_recall
    plane_recall_new[:, 1] = gt_plane_num
    plane_recall_new[:, 2] = plane_num

    return plane_recall_new, pixel_recall

# https://github.com/svip-lab/PlanarReconstruction/blob/master/utils/metric.py
# https://github.com/yi-ming-qian/interplane/blob/master/utils/metric.py
def evaluateMasks(predSegmentations, gtSegmentations, device, pred_non_plane_idx, gt_non_plane_idx=20, printInfo=False):
    """
    :param predSegmentations:
    :param gtSegmentations:
    :param device:
    :param pred_non_plane_idx:
    :param gt_non_plane_idx:
    :param printInfo:
    :return:
    """
    predSegmentations = torch.from_numpy(predSegmentations).to(device)
    gtSegmentations = torch.from_numpy(gtSegmentations).to(device)

    pred_masks = []
    if pred_non_plane_idx > 0:
        for i in range(pred_non_plane_idx):
            mask_i = predSegmentations == i
            mask_i = mask_i.float()
            if mask_i.sum() > 0:
                pred_masks.append(mask_i)
    else:
        assert pred_non_plane_idx == -1 or pred_non_plane_idx == 0
        for i in range(gt_non_plane_idx + 1, 100):
            mask_i = predSegmentations == i
            mask_i = mask_i.float()
            if mask_i.sum() > 0:
                pred_masks.append(mask_i)
    predMasks = torch.stack(pred_masks, dim=0)

    gt_masks = []
    if gt_non_plane_idx > 0:
        for i in range(gt_non_plane_idx):
            mask_i = gtSegmentations == i
            mask_i = mask_i.float()
            if mask_i.sum() > 0:
                gt_masks.append(mask_i)
    else:
        assert pred_non_plane_idx == -1 or pred_non_plane_idx == 0
        for i in range(gt_non_plane_idx+1, 100):
            mask_i = gtSegmentations == i
            mask_i = mask_i.float()
            if mask_i.sum() > 0:
                gt_masks.append(mask_i)
    gtMasks = torch.stack(gt_masks, dim=0)

    valid_mask = (gtMasks.max(0)[0]).unsqueeze(0)

    gtMasks = torch.cat([gtMasks, torch.clamp(1 - gtMasks.sum(0, keepdim=True), min=0)], dim=0)  # M+1, H, W
    predMasks = torch.cat([predMasks, torch.clamp(1 - predMasks.sum(0, keepdim=True), min=0)], dim=0)  # N+1, H, W

    intersection = (gtMasks.unsqueeze(1) * predMasks * valid_mask).sum(-1).sum(-1).float()
    union = (torch.max(gtMasks.unsqueeze(1), predMasks) * valid_mask).sum(-1).sum(-1).float()

    N = intersection.sum()

    RI = 1 - ((intersection.sum(0).pow(2).sum() + intersection.sum(1).pow(2).sum()) / 2 - intersection.pow(2).sum()) / (
            N * (N - 1) / 2)
    joint = intersection / N
    marginal_2 = joint.sum(0)
    marginal_1 = joint.sum(1)
    H_1 = (-marginal_1 * torch.log2(marginal_1 + (marginal_1 == 0).float())).sum()
    H_2 = (-marginal_2 * torch.log2(marginal_2 + (marginal_2 == 0).float())).sum()

    B = (marginal_1.unsqueeze(-1) * marginal_2)
    log2_quotient = torch.log2(torch.clamp(joint, 1e-8) / torch.clamp(B, 1e-8)) * (torch.min(joint, B) > 1e-8).float()
    MI = (joint * log2_quotient).sum()
    voi = H_1 + H_2 - 2 * MI

    IOU = intersection / torch.clamp(union, min=1)
    SC = ((IOU.max(-1)[0] * torch.clamp((gtMasks * valid_mask).sum(-1).sum(-1), min=1e-4)).sum() / N + (
            IOU.max(0)[0] * torch.clamp((predMasks * valid_mask).sum(-1).sum(-1), min=1e-4)).sum() / N) / 2
    info = [RI.item(), voi.item(), SC.item()]
    if printInfo:
        print('mask statistics', info)
        pass
    return info
