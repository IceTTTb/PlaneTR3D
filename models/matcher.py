# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 2, cost_param: float = 1, cost_center: float = 2, cost_emb: float = 0.5):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_param: This is the relative weight of the error of plane parameters in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_param = cost_param
        self.cost_center = cost_center
        self.cost_emb = cost_emb
        assert cost_class != 0 or cost_param != 0, "all costs can not be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, targets_emb=None):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, 2] with the classification logits
                 "pred_param": Tensor of dim [batch_size, num_queries, 3] with the predicted plane parameters

            targets: This is a dict that contains at least these entries:
                 "labels": tensor of dim [batch_size, num_target_planes, 1]
                 "params": Tensor of dim [batch_size, num_target_planes, 3] containing the target plane parameters

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_planes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, 3]
        out_param = outputs["pred_param"].flatten(0, 1)  # [batch_size * num_queries, 3]
        # print('******-------------', out_prob.max(), out_prob.min(), out_param.max(), out_param.min())

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([tgt[:, 0] for tgt in targets]).long()  # [batch_size * num_target_planes]
        tgt_param = torch.cat([tgt[:, 1:4] for tgt in targets])  # [batch_size * num_target_planes, 3]


        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between params
        cost_param = torch.cdist(out_param, tgt_param, p=1)  # batch_size * num_queries * batch_size * num_target_planes

        # Compute the L2 cost between centers
        if 'pred_center' in outputs.keys():
            out_center = outputs["pred_center"].flatten(0, 1)  # [batch_size * num_queries, 2]
            tgt_center = torch.cat([tgt[:, 4:6] for tgt in targets])  # [batch_size * num_target_planes, 2]
            cost_center = torch.cdist(out_center, tgt_center, p=2)  # batch_size * num_queries * batch_size * num_target_planes
        else:
            cost_center = 0.

        if targets_emb is not None:
            out_emb = outputs['pred_plane_embedding'].flatten(0, 1)  # [batch_size * num_queries, c_emb]
            tgt_emb = torch.cat([tgt[:, :] for tgt in targets_emb])  # [batch_size * num_target_planes, c_emb]
            # Compute the L1 cost between embs
            cost_emb = torch.cdist(out_emb, tgt_emb, p=2)  # batch_size * num_queries * batch_size * num_target_planes
        else:
            cost_emb = 0.

        # Final cost matrix
        # print('max', cost_param.max(), cost_class.max(), cost_center.max())
        # print('min', cost_param.min(), cost_class.min(), cost_center.min())
        # print('mean', cost_param.mean(), cost_class.mean(), cost_center.mean())
        # exit()
        C = self.cost_param * cost_param + self.cost_class * cost_class + self.cost_center * cost_center + self.cost_emb * cost_emb
        C = C.view(bs, num_queries, -1).cpu()

        # print(cost_param.max(), cost_param.min(), cost_class.max(), cost_class.min(), cost_center.max(), cost_center.min())

        sizes = [tgt.shape[0] for tgt in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        res_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        # print("tgt_ids shape/value: ", tgt_ids.shape, tgt_ids)
        # print("")
        # print("cost_class.shape = ", cost_class.shape)
        # print("cost_param.shape = ", cost_param.shape)
        # print("C.shape = ", C.shape)
        # print("sizes = ", sizes)
        # print('*'*10)
        # print("res indices = ")
        # for i in range(len(res_indices)):
        #     print(res_indices[i])
        # print('*' * 10)
        # exit()

        # import pdb
        # pdb.set_trace()

        return res_indices

class HungarianMatcher_DEBUG(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 2, cost_param: float = 1, cost_center: float = 2):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_param: This is the relative weight of the error of plane parameters in the matching cost
        """
        super().__init__()
        self.cost_class = 1.0  # cost_class
        self.cost_param = 1.0  # cost_param
        self.cost_center = 4.0  # cost_center
        assert cost_class != 0 or cost_param != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, 2] with the classification logits
                 "pred_param": Tensor of dim [batch_size, num_queries, 3] with the predicted plane parameters

            targets: This is a dict that contains at least these entries:
                 "labels": tensor of dim [batch_size, num_target_planes, 1]
                 "params": Tensor of dim [batch_size, num_target_planes, 3] containing the target plane parameters

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_planes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, 3]
        out_param = outputs["pred_param"].flatten(0, 1)  # [batch_size * num_queries, 3]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([tgt[:, 0] for tgt in targets]).long()  # [batch_size * num_target_planes]
        tgt_param = torch.cat([tgt[:, 1:4] for tgt in targets])  # [batch_size * num_target_planes, 3]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between params
        cost_param = torch.cdist(out_param, tgt_param, p=1)  # batch_size * num_queries * batch_size * num_target_planes

        # Compute the L2 cost between centers
        if 'pred_center' in outputs.keys():
            out_center = outputs["pred_center"].flatten(0, 1)  # [batch_size * num_queries, 2]
            tgt_center = torch.cat([tgt[:, 4:] for tgt in targets])  # [batch_size * num_target_planes, 2]
            cost_center = torch.cdist(out_center, tgt_center, p=2)  # batch_size * num_queries * batch_size * num_target_planes
        else:
            cost_center = 0.

        # Final cost matrix
        # print('max', cost_param.max(), cost_class.max(), cost_center.max())
        # print('min', cost_param.min(), cost_class.min(), cost_center.min())
        # print('mean', cost_param.mean(), cost_class.mean(), cost_center.mean())
        # exit()
        C = self.cost_param * cost_param + self.cost_class * cost_class + self.cost_center * cost_center
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [tgt.shape[0] for tgt in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        res_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        # print("tgt_ids shape/value: ", tgt_ids.shape, tgt_ids)
        # print("")
        # print("cost_class.shape = ", cost_class.shape)
        # print("cost_param.shape = ", cost_param.shape)
        # print("C.shape = ", C.shape)
        # print("sizes = ", sizes)
        # print('*'*10)
        # print("res indices = ")
        # for i in range(len(res_indices)):
        #     print(res_indices[i])
        # print('*' * 10)
        # exit()

        return res_indices