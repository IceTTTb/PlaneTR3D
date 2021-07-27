import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os

from models.position_encoding import build_position_encoding
from models.transformer import build_transformer
from models.HRNet import build_hrnet

logger = logging.getLogger(__name__)
use_biase = False
use_align_corners = False


def conv_bn_relu(in_dim, out_dim, k=1, pad=0):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, (k, k), padding=pad, bias=use_biase),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(inplace=True)
    )


class featureFusionNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(featureFusionNet, self).__init__()

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=use_biase),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=use_biase),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.init_weights()

    def init_weights(self):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, f1, f2, f3, f4, ori_h, ori_w):
        out_size = (ori_h//16, ori_w//16)
        f1_scaled = F.upsample(f1, size=out_size, mode='bilinear', align_corners=use_align_corners)
        f2_scaled = F.upsample(f2, size=out_size, mode='bilinear', align_corners=use_align_corners)
        f3_scaled = F.upsample(f3, size=out_size, mode='bilinear', align_corners=use_align_corners)
        f4_scaled = F.upsample(f4, size=out_size, mode='bilinear', align_corners=use_align_corners)

        f_cat = torch.cat((f1_scaled, f2_scaled, f3_scaled, f4_scaled), dim=1)

        f_out = self.fusion(f_cat)

        return f_out


class top_down(nn.Module):
    def __init__(self, in_channels=[], channel=64, m_dim=256, double_upsample=False):
        super(top_down, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.double_upsample = double_upsample

        # top down
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=use_align_corners)
        if double_upsample:
            self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=use_align_corners)
        self.up_conv3 = conv_bn_relu(channel, channel, 1)
        self.up_conv2 = conv_bn_relu(channel, channel, 1)
        self.up_conv1 = conv_bn_relu(channel, channel, 1)

        # lateral
        self.c4_conv = conv_bn_relu(in_channels[3], channel, 1)
        self.c3_conv = conv_bn_relu(in_channels[2], channel, 1)
        self.c2_conv = conv_bn_relu(in_channels[1], channel, 1)
        self.c1_conv = conv_bn_relu(in_channels[0], channel, 1)

        self.m_conv_dict = nn.ModuleDict({})
        self.m_conv_dict['m4'] = conv_bn_relu(m_dim, channel)

        self.p0_conv = nn.Conv2d(channel, channel, (3, 3), padding=1)

    def init_weights(self):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, memory):
        c1, c2, c3, c4 = x

        p4 = self.c4_conv(c4) + self.m_conv_dict['m4'](memory)

        p3 = self.up_conv3(self.upsample(p4)) + self.c3_conv(c3)

        p2 = self.up_conv2(self.upsample(p3)) + self.c2_conv(c2)

        p1 = self.up_conv1(self.upsample(p2)) + self.c1_conv(c1)

        if self.double_upsample:
            p0 = self.upsample2(p1)
        else:
            p0 = self.upsample(p1)

        p0 = self.relu(self.p0_conv(p0))

        return p0, p1, p2, p3, p4


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def get_lines_features(feat, pos, lines, size_ori, n_pts=21):
    """
    :param feat: B, C, H, W
    :param lines: B, N, 4
    :return: B, C, N
    """
    ho, wo = size_ori
    b, c, hf, wf = feat.shape
    line_num = lines.shape[1]

    with torch.no_grad():
        scale_h = ho / hf
        scale_w = wo / wf
        scaled_lines = lines.clone()
        scaled_lines[:, :, 0] = scaled_lines[:, :, 0] / scale_w
        scaled_lines[:, :, 1] = scaled_lines[:, :, 1] / scale_h
        scaled_lines[:, :, 2] = scaled_lines[:, :, 2] / scale_w
        scaled_lines[:, :, 3] = scaled_lines[:, :, 3] / scale_h

        spts, epts = torch.split(scaled_lines, (2, 2), dim=-1)  # B, N, 2

        if n_pts > 2:
            delta_pts = (epts - spts) / (n_pts-1)  # B, N, 2
            delta_pts = delta_pts.unsqueeze(dim=2).expand(b, line_num, n_pts, 2)  # b, n, n_pts, 2
            steps = torch.linspace(0., n_pts-1, n_pts).view(1, 1, n_pts, 1).to(device=lines.device)

            spts_expand = spts.unsqueeze(dim=2).expand(b, line_num, n_pts, 2)  # b, n, n_pts, 2
            line_pts = spts_expand + delta_pts * steps  # b, n, n_pts, 2

        elif n_pts == 2:
            line_pts = torch.stack([spts, epts], dim=2)  # b, n, n_pts, 2
        elif n_pts == 1:
            line_pts = torch.cat((spts, epts), dim=1).unsqueeze(dim=2)

        line_pts[:, :, :, 0] = line_pts[:, :, :, 0] / float(wf-1) * 2. - 1.
        line_pts[:, :, :, 1] = line_pts[:, :, :, 1] / float(hf-1) * 2. - 1.

        line_pts = line_pts.detach()

    sample_feats = F.grid_sample(feat, line_pts)  # b, c, n, n_pts

    b, c, ln, pn = sample_feats.shape
    sample_feats = sample_feats.permute(0, 1, 3, 2).contiguous().view(b, -1, ln)

    sample_pos = F.grid_sample(pos, line_pts)
    sample_pos = torch.mean(sample_pos, dim=-1)

    return sample_feats, sample_pos  # b, c, n


class PlaneTR_HRNet(nn.Module):
    def __init__(self, cfg, position_embedding_mode='sine'):
        super(PlaneTR_HRNet, self).__init__()
        num_queries = cfg.model.num_queries
        plane_embedding_dim = cfg.model.plane_embedding_dim
        loss_layer_num = cfg.model.loss_layer_num
        predict_center = cfg.model.if_predict_center
        use_lines = cfg.model.use_lines

        # Feature extractor
        self.backbone = build_hrnet(cfg.model)
        self.backbone_channels = self.backbone.out_channels

        # pre-defined
        self.loss_layer_num = loss_layer_num
        assert self.loss_layer_num < 6
        self.return_inter = False
        self.predict_center = predict_center
        self.use_lines = use_lines
        self.num_sample_pts = cfg.model.num_sample_pts
        self.if_predict_depth = cfg.model.if_predict_depth
        assert cfg.model.stride == 1

        self.hidden_dim = 256
        self.num_queries = num_queries
        self.context_channels = self.backbone_channels[-1]
        self.line_channels = self.backbone_channels[1]
        if self.num_sample_pts <= 2:
            self.lines_reduce = nn.Sequential(
                nn.Linear(self.hidden_dim * self.num_sample_pts, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.hidden_dim), )
        else:
            self.lines_reduce = nn.Sequential(
                nn.MaxPool1d(8, 8),
                nn.Linear(self.hidden_dim * self.num_sample_pts // 8, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.hidden_dim), )

        self.plane_embedding_dim = plane_embedding_dim
        self.channel = 64

        # Transformer Branch
        self.input_proj = nn.Conv2d(self.context_channels, self.hidden_dim, kernel_size=1)
        if use_lines:
            self.lines_proj = nn.Conv2d(self.line_channels, self.hidden_dim, kernel_size=1)
        self.position_embedding = build_position_encoding(position_embedding_mode, hidden_dim=self.hidden_dim)
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        self.transformer = build_transformer(hidden_dim=self.hidden_dim, dropout=0.1, nheads=8, dim_feedforward=1024,
                                             enc_layers=6, dec_layers=cfg.model.dec_layers, pre_norm=True, return_inter=self.return_inter,
                                             use_lines=use_lines, loss_layer_num=self.loss_layer_num)
        # instance-level plane embedding
        self.plane_embedding = MLP(self.hidden_dim, self.hidden_dim, self.plane_embedding_dim, 3)
        # plane / non-plane classifier
        self.plane_prob = nn.Linear(self.hidden_dim, 2 + 1)
        # instance-level plane 3D parameters
        self.plane_param = MLP(self.hidden_dim, self.hidden_dim, 3, 3)
        # instance-level plane center
        if self.predict_center:
            self.plane_center = MLP(self.hidden_dim, self.hidden_dim, 2, 3)

        # Convolution Branch
        # top_down
        self.top_down = top_down(self.backbone_channels, self.channel, m_dim=self.hidden_dim, double_upsample=False)
        # pixel embedding
        self.pixel_embedding = nn.Conv2d(self.channel, self.plane_embedding_dim, (1, 1), padding=0)
        # pixel-level plane center
        if self.predict_center:
            self.pixel_plane_center = nn.Conv2d(self.channel, 2, (1, 1), padding=0)
        # pixel-level depth
        if self.if_predict_depth:
            self.top_down_depth = top_down(self.backbone_channels, self.channel, m_dim=self.hidden_dim, double_upsample=False)
            self.depth = nn.Conv2d(self.channel, 1, (1, 1), padding=0)

    def forward(self, x, lines=None, num_lines=None, scale=None):
        bs, _, ho, wo = x.shape
        c1, c2, c3, c4 = self.backbone(x)  # c: 32, 64, 128, 256

        # context src feature
        src = c4

        # context feature proj
        src = self.input_proj(src)  # b, hidden_dim, h, w

        # position embedding
        pos = self.position_embedding(src)

        # transformer
        if self.use_lines:
            assert lines is not None
            # line src feature
            src_line = c2

            # line feature proj
            line_proj = self.lines_proj(src_line)  # b, hidden_dim, h', w'

            # position embedding
            pos_line = self.position_embedding(line_proj)

            # line feature sampling, lines: b, max_line_num, hidden_dim
            lines_feat, lines_pos = get_lines_features(line_proj, pos_line, lines, [ho, wo], self.num_sample_pts)  # B, hidden_dim, max_line_num

            lines_feat = lines_feat.permute(0, 2, 1)  # B, max_line_num, hidden_dim
            lines_pos = lines_pos.permute(0, 2, 1)  # B, max_line_num, hidden_dim

            # get line mask: bs, max_line_num
            max_line_num = lines.shape[1]
            masks = []
            for i in range(bs):
                valid_line_num = int(num_lines[i])
                mask_i = torch.zeros((1, max_line_num), dtype=torch.bool, device=lines.device)  # 1, max_line_num
                mask_i[:, valid_line_num:] = True
                masks.append(mask_i)
            mask_lines = torch.cat(masks, dim=0).contiguous()  # b, max_line_num

            if self.lines_reduce is not None:
                lines_feat = self.lines_reduce(lines_feat)

            mask_temp = mask_lines.unsqueeze(-1).float()
            lines_feat = lines_feat * (1 - mask_temp)  # B, max_line_num, hidden_dim
            lines_pos = lines_pos * (1 - mask_temp)  # B, max_line_num, hidden_dim

            hs_all, _, memory = self.transformer(src, self.query_embed.weight, pos, tgt=None,
                                                       src_lines=lines_feat, mask_lines=mask_lines,
                                                       pos_embed_lines=lines_pos)  # memory: b, c, h, w
        else:
            hs_all, _, memory = self.transformer(src, self.query_embed.weight, pos, tgt=None,
                                                       src_lines=None, mask_lines=None,
                                                       pos_embed_lines=None)  # memory: b, c, h, w

        # ------------------------------------------------------- plane instance decoder
        hs = hs_all[-self.loss_layer_num:, :, :, :].contiguous()  # dec_layers, b, num_queries, hidden_dim
        # plane embedding
        plane_embedding = self.plane_embedding(hs)  # 1, b, num_queries, 2
        # plane classifier
        plane_prob = self.plane_prob(hs)  # 1, b, num_queries, 3
        # plane parameters
        plane_param = self.plane_param(hs)  # 1, b, num_queries, 3
        # plane center
        if self.predict_center:
            plane_center = self.plane_center(hs)  # 1, b, num_queries, 2
            plane_center = torch.sigmoid(plane_center)

        # --------------------------------------------------- pixel-level decoder
        p0, p1, p2, p3, p4 = self.top_down((c1, c2, c3, c4), memory)
        pixel_embedding = self.pixel_embedding(p0)  # b, 2, h, w
        if self.predict_center:
            pixel_center = self.pixel_plane_center(p0)  # b, 2, h, w
            pixel_center = torch.sigmoid(pixel_center)  # 0~1
        if self.if_predict_depth:
            p_depth, _, _, _, _ = self.top_down_depth((c1, c2, c3, c4), memory)
            pixel_depth = self.depth(p_depth)

        # ------------------------------------------------------output
        output = {'pred_logits': plane_prob[-1], 'pred_param': plane_param[-1],
                  'pred_plane_embedding': plane_embedding[-1], 'pixel_embedding': pixel_embedding}
        if self.predict_center:
            output['pixel_center'] = pixel_center
            output['pred_center'] = plane_center[-1]
        if self.if_predict_depth:
            output['pixel_depth'] = pixel_depth
        if self.loss_layer_num > 1 and self.training:
            assert self.loss_layer_num == 3
            assert plane_prob.shape[0] == 3
            aux_outputs = []
            for i in range(self.loss_layer_num - 1):
                aux_1 = {'pred_logits': plane_prob[i], 'pred_plane_embedding': plane_embedding[i],
                         'pixel_embedding': pixel_embedding}
                if self.predict_center:
                    aux_1['pred_center'] = plane_center[i]
                aux_outputs.append(aux_1)
            output['aux_outputs'] = aux_outputs
        return output
