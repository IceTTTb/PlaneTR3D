import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
import numpy as np

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, k_inv_dot_xy1):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.k_inv_dot_xy1 = k_inv_dot_xy1  # 3, hw
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        # empty_weight[-2] = 3

        self.register_buffer('empty_weight', empty_weight)


    def loss_labels(self, outputs, targets, indices, num_planes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([tgt[:, 0][J].long() for tgt, (_, J) in zip (targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        # print("idx = ", idx)
        # print("target_classes.shape = ", target_classes.shape)
        # print("target_classes = ", target_classes)
        # print("src_logits.shape = ", src_logits.shape)
        # print("empty_weight = ", self.empty_weight)
        # exit()

        ##################### 2020.2.28
        # loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight.cuda(), ignore_index=0)
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight.cuda())

        # import pdb
        # pdb.set_trace()

        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_planes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([tgt.shape[0] for tgt in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_param(self, outputs, targets, indices, num_planes, log=True):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_param' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_param = outputs['pred_param'][idx]  # N, 3
        target_param = torch.cat([tgt[:, 1:4][i] for tgt, (_, i) in zip(targets, indices)], dim=0)

        # l1 loss
        loss_param_l1 = torch.mean(torch.sum(torch.abs(target_param - src_param), dim=1))

        # cos loss
        similarity = torch.nn.functional.cosine_similarity(src_param, target_param, dim=1)  # N
        loss_param_cos = torch.mean(1-similarity)
        angle = torch.mean(torch.acos(torch.clamp(similarity, -1, 1)))

        losses = {}
        losses['loss_param_l1'] = loss_param_l1
        losses['loss_param_cos'] = loss_param_cos
        if log:
            losses['mean_angle'] = angle * 180.0 / np.pi

        return losses

    def loss_center(self, outputs, targets, indices, num_planes, log=True):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_center' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_center = outputs['pred_center'][idx]  # N, 2
        target_center = torch.cat([tgt[:, 4:6][i] for tgt, (_, i) in zip(targets, indices)], dim=0)
        # print('src center = \n', src_center.detach().cpu().numpy())
        # print('tgt_center = \n', target_center.detach().cpu().numpy())
        # exit()

        # l1 loss
        delta_xy = torch.abs(target_center - src_center)  # N, 2
        dist = torch.norm(delta_xy, dim=-1)  # N
        loss_center_l2 = torch.mean(dist)

        losses = {}
        losses['loss_center_instance'] = loss_center_l2

        '''
        pixel_center = outputs['pixel_center']  # b, 2, h, w
        b, _, h, w = pixel_center.shape
        assert b == len(targets)

        loss_center_l2_pixel = 0.
        for bi in range(b):
            indices_bi = indices[bi]
            idx_out = indices_bi[0]
            idx_tgt = indices_bi[1]

            segmentation = outputs['gt_instance_map'][bi]  # 21, h, w
            cur_pxiel_center = pixel_center[bi]  # 2, h, w
            loss_bi = 0.
            for pi in range(num_planes):
                gt_plane_idx = int(idx_tgt[pi])
                mask = segmentation[gt_plane_idx, :, :].view(1, h, w)
                mask = mask > 0
                centers = torch.masked_select(cur_pxiel_center, mask).view(2, -1)  # 2, n
                gt_center = targets[bi][gt_plane_idx, 4:6].contiguous().view(2, 1)
                loss_dist = torch.norm(torch.abs(centers - gt_center), dim=0).mean()
                loss_bi += loss_dist
            loss_bi = loss_bi / num_planes
            loss_center_l2_pixel += loss_bi
        loss_center_l2_pixel = loss_center_l2_pixel / bi
        losses['loss_center_pixel'] = loss_center_l2_pixel
        '''
        if 'gt_plane_pixel_centers' in outputs.keys():
            gt_plane_pixel_centers = outputs['gt_plane_pixel_centers']
            pixel_center = outputs['pixel_center']  # b, 2, h, w
            valid_region = outputs['valid_region']  # b, 1, h, w
            mask = valid_region > 0
            pixel_dist = torch.norm(torch.abs(gt_plane_pixel_centers - pixel_center), dim=1, keepdim=True)  #b, 1, h, w
            loss_pixel_center = torch.mean(pixel_dist[mask])
            losses['loss_center_pixel'] = loss_pixel_center

        return losses

    def loss_embedding(self, outputs, targets, indices, num_planes_sum, log=True, t_pull=0.5, t_push=1.5):
        embedding_pixel = outputs['pixel_embedding']  # b, c, h, w
        embedding_instance = outputs['pred_plane_embedding']  # b, num_query, c
        b, c, h, w = embedding_pixel.shape

        assert b == len(targets)

        pull_losses = 0.
        push_losses = 0.
        losses = 0.

        for bi in range(b):
            embedding = embedding_pixel[bi, :, :, :].contiguous()
            num_planes = targets[bi].shape[0]
            device = embedding.device

            indices_bi = indices[bi]
            idx_out = indices_bi[0]
            idx_tgt = indices_bi[1]

            assert idx_tgt.max()+1 == num_planes

            segmentation = outputs['gt_instance_map'][bi]  # 21, h, w

            embeddings = []
            centers = []
            # select embedding with segmentation
            for i in range(num_planes):
                gt_plane_idx = int(idx_tgt[i])
                mask = segmentation[gt_plane_idx, :, :].view(1, h, w)
                mask = mask > 0
                feature = torch.transpose(torch.masked_select(embedding, mask).view(c, -1),
                                          0, 1)
                embeddings.append(feature)  # plane_pt_num, c

                pred_plane_idx = int(idx_out[i])
                center = embedding_instance[bi, pred_plane_idx, :].contiguous().view(1, c)
                centers.append(center)

            # intra-embedding loss within a plane
            pull_loss = torch.Tensor([0.0]).to(device)
            for feature, center in zip(embeddings, centers):
                # l2 dist
                dis = torch.norm(feature - center, 2, dim=1) - t_pull
                dis = F.relu(dis)
                pull_loss += torch.mean(dis)

                # cos dist
                # dis_cos = 1 - F.cosine_similarity(feature, center, dim=1) - 0.01519
                # dis_cos = F.relu(dis_cos)
                # pull_loss += torch.mean(dis_cos)

                # print(torch.mean(dis))
                # print(torch.mean(dis_cos))

            pull_loss /= int(num_planes)

            if num_planes == 1:
                losses += pull_loss
                pull_losses += pull_loss
                push_losses += 0.
                continue

            # inter-plane loss
            centers = torch.cat(centers, dim=0)  # n, c
            A = centers.repeat(1, int(num_planes)).view(-1, c)
            B = centers.repeat(int(num_planes), 1)
            distance = torch.norm(A - B, 2, dim=1).view(int(num_planes), int(num_planes))
            # distance_cos = 1 - F.cosine_similarity(A, B, dim=1).view(int(num_planes), int(num_planes))

            # select pair wise distance from distance matrix
            eye = torch.eye(int(num_planes)).to(device)
            pair_distance = torch.masked_select(distance, eye == 0)
            # pair_distance_cos = torch.masked_select(distance_cos, eye == 0)

            # import pdb
            # pdb.set_trace()

            # l2 dist
            pair_distance = t_push - pair_distance
            pair_distance = F.relu(pair_distance)
            push_loss = torch.mean(pair_distance).view(-1)

            # cos dist
            # pair_distance_cos = 1.0 - pair_distance_cos
            # pair_distance_cos = F.relu(pair_distance_cos)
            # push_loss += torch.mean(pair_distance_cos).view(-1)

            loss = pull_loss + push_loss

            losses += loss
            pull_losses += pull_loss
            push_losses += push_loss

        losses_dict = {}
        losses_dict['loss_embedding'] = losses / float(b)
        if log:
            losses_dict['loss_pull'] = pull_losses / float(b)
            losses_dict['loss_push'] = push_losses / float(b)

        return losses_dict

    def loss_Q(self, outputs, targets, indices, num_planes_sum, log=True):
        gt_depths = outputs['gt_depth']  # b, 1, h, w
        b, _, h, w = gt_depths.shape

        assert b == len(targets)

        losses = 0.

        for bi in range(b):
            num_planes = targets[bi].shape[0]
            segmentation = outputs['gt_instance_map'][bi]  # 21, h, w
            device = segmentation.device

            depth = gt_depths[bi]  # 1, h, w
            k_inv_dot_xy1_map = (self.k_inv_dot_xy1).clone().view(3, h, w).to(device)
            gt_pts_map = k_inv_dot_xy1_map * depth  # 3, h, w

            indices_bi = indices[bi]
            idx_out = indices_bi[0]
            idx_tgt = indices_bi[1]
            assert idx_tgt.max() + 1 == num_planes

            # select pixel with segmentation
            loss_bi = 0.
            for i in range(num_planes):
                gt_plane_idx = int(idx_tgt[i])
                mask = segmentation[gt_plane_idx, :, :].view(1, h, w)
                mask = mask > 0

                pts = torch.masked_select(gt_pts_map, mask).view(3, -1)  # 3, plane_pt_num

                pred_plane_idx = int(idx_out[i])
                param = outputs['pred_param'][bi][pred_plane_idx].view(1, 3)
                # param = targets[bi][gt_plane_idx, 1:].view(1, 3)

                #########################################
                # param_gt = targets[bi][gt_plane_idx, 1:4].view(1, 3)
                # gt_err = torch.mean(torch.abs(torch.matmul(param_gt, pts) - 1))  # 1, plane_pt_num
                # print(gt_err)
                #########################################

                loss = torch.abs(torch.matmul(param, pts) - 1)  # 1, plane_pt_num
                loss = loss.mean()
                loss_bi += loss
            loss_bi = loss_bi / float(num_planes)
            losses += loss_bi

            # exit()

        losses_dict = {}
        losses_dict['loss_Q'] = losses / float(b)

        return losses_dict

    def loss_depth(self, outputs, targets, indices, num_planes_sum, log=True):
        gt_pixel_depth = outputs['gt_depth']
        pred_pixel_depth = outputs['pixel_depth']

        mask = (gt_pixel_depth > 1e-4).float()
        loss = torch.sum(torch.abs(pred_pixel_depth - gt_pixel_depth) * mask) / torch.clamp(mask.sum(), min=1)

        losses = {'loss_depth_pixel': loss}

        # import pdb
        # pdb.set_trace()

        if 'final_depth' in outputs.keys():
            if 'final_depth' in outputs.keys():
                pred_final_depth = outputs['final_depth']
                loss_final_depth = torch.sum(torch.abs(pred_final_depth - gt_pixel_depth) * mask) / torch.clamp(
                    mask.sum(), min=1)
                losses['loss_final_depth'] = loss_final_depth

            if 'final_depth_ref' in outputs.keys():
                pred_final_depth_ref = outputs['final_depth_ref']
                loss_final_depth_ref = torch.sum(torch.abs(pred_final_depth_ref - gt_pixel_depth) * mask) / torch.clamp(
                    mask.sum(), min=1)
                losses['loss_final_depth_ref'] = loss_final_depth_ref

        return losses

    def loss_prob_pixel(self, outputs, targets, indices, num_planes_sum, log=True):
        gamma = 2.
        alpha = 0.25

        gt_semantic = outputs['gt_semantic']  # b, 1, h, w
        pred_pixel_plane_prob = outputs['pixel_plane_prob']  # b, 1, h, w
        pred_pixel_plane_prob = torch.sigmoid(pred_pixel_plane_prob)  # b, 1, h, w

        loss = - alpha * (1 - pred_pixel_plane_prob) ** gamma * gt_semantic * torch.log(pred_pixel_plane_prob) \
               - (1 - alpha) * pred_pixel_plane_prob ** gamma * (1 - gt_semantic) * torch.log(1 - pred_pixel_plane_prob)

        loss = torch.mean(loss)

        losses = {'loss_prob_pixel': loss}

        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])

        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_planes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'param': self.loss_param,
            'loss_cardinality': self.loss_cardinality,
            'embedding': self.loss_embedding,
            'Q': self.loss_Q,
            'center': self.loss_center,
            'depth': self.loss_depth,
            'prob_pixel': self.loss_prob_pixel
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_planes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_planes = sum(tgt.shape[0] for tgt in targets)
        num_planes = torch.as_tensor([num_planes], dtype=torch.float, device=next(iter(outputs.values())).device)

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_planes)
        num_planes = torch.clamp(num_planes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            # print(loss)
            losses.update(self.get_loss(loss, outputs, targets, indices, num_planes))
            # print(loss, 'end-', '*'*10)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        losses_aux = []

        if 'aux_outputs' in outputs.keys():
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                losses_aux_i = {}
                # print(aux_outputs.keys())
                # continue
                # indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    # if 'embedding' in loss:
                    #     continue
                    if 'param' in loss or 'Q' in loss or 'depth' in loss: #or 'embedding' in loss:
                        continue
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_planes, **kwargs)
                    # l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses_aux_i.update(l_dict)
                losses_aux.append(losses_aux_i)

        return losses, indices, losses_aux