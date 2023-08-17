
from .feature_net import build_feature_net
from .layer import *
from config import net_config as config

import copy
from torch.nn.parallel import data_parallel
from scipy.stats import norm


class RpnHead(nn.Module):
    def __init__(self, config, in_channels=128):
        super(RpnHead, self).__init__()
        self.drop = nn.Dropout3d(p=0.5, inplace=False)
        self.conv = nn.Sequential(nn.Conv3d(in_channels, 64, kernel_size=1),
                                  nn.ReLU())

        self.logits = nn.Conv3d(64, 1 * len(config['anchors']), kernel_size=1)
        self.deltas = nn.Conv3d(64, 6 * len(config['anchors']), kernel_size=1)

    def forward(self, f):
        # out = self.drop(f)
        out = self.conv(f)

        logits = self.logits(out)
        deltas = self.deltas(out)
        size = logits.size()
        logits = logits.view(logits.size(0), logits.size(1), -1)
        logits = logits.transpose(1, 2).contiguous().view(
            size[0], size[2], size[3], size[4], len(config['anchors']), 1)

        size = deltas.size()
        deltas = deltas.view(deltas.size(0), deltas.size(1), -1)
        deltas = deltas.transpose(1, 2).contiguous().view(
            size[0], size[2], size[3], size[4], len(config['anchors']), 6)

        return logits, deltas


class RcnnHead(nn.Module):
    def __init__(self, cfg, in_channels=128):
        super(RcnnHead, self).__init__()
        self.num_class = cfg['num_class']
        self.crop_size = cfg['rcnn_crop_size']

        self.fc1 = nn.Linear(
            in_channels * self.crop_size[0] * self.crop_size[1] * self.crop_size[2], 512)
        self.fc2 = nn.Linear(512, 256)
        self.logit = nn.Linear(256, self.num_class)
        self.delta = nn.Linear(256, self.num_class * 6)

    def forward(self, crops):
        x = crops.view(crops.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        # x = F.dropout(x, 0.5, training=self.training)
        logits = self.logit(x)
        deltas = self.delta(x)

        return logits, deltas


class CropRoi(nn.Module):
    def __init__(self, cfg):
        super(CropRoi, self).__init__()
        self.cfg = cfg
        self.rcnn_crop_size = cfg['rcnn_crop_size']
        self.scale = cfg['stride']
        self.DEPTH, self.HEIGHT, self.WIDTH = cfg['crop_size']

    def forward(self, f, inputs, proposals):
        self.DEPTH, self.HEIGHT, self.WIDTH = inputs.shape[2:]

        crops = []
        for p in proposals:
            b = int(p[0])
            center = p[2:5]
            side_length = p[5:8]
            c0 = center - side_length / 2  # left bottom corner
            c1 = c0 + side_length  # right upper corner
            c0 = (c0 / self.scale).floor().long()
            c1 = (c1 / self.scale).ceil().long()
            minimum = torch.LongTensor([[0, 0, 0]]).cuda()
            maximum = torch.LongTensor(
                np.array([[self.DEPTH, self.HEIGHT, self.WIDTH]]) / self.scale).cuda()

            c0 = torch.cat((c0.unsqueeze(0).cuda(), minimum), 0)
            c1 = torch.cat((c1.unsqueeze(0).cuda(), maximum), 0)
            c0, _ = torch.max(c0, 0)
            c1, _ = torch.min(c1, 0)

            # Slice 0 dim, should never happen
            if np.any((c1 - c0).cpu().data.numpy() < 1):
                print(p)
                print('c0:', c0, ', c1:', c1)
            crop = f[b, :, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]]
            crop = F.adaptive_max_pool3d(crop, self.rcnn_crop_size)
            crops.append(crop)

        crops = torch.stack(crops)

        return crops


class MainNet(nn.Module):
    def __init__(self, config, mode='train'):
        super(MainNet, self).__init__()
        self.cfg = config
        self.mode = mode

        self.feature_net = build_feature_net()

        self.rpn = RpnHead(config, in_channels=128)

        self.rcnn_head = RcnnHead(config, in_channels=64)
        self.rcnn_crop = CropRoi(config)

        self.feature_size = None

    def forward(self, inputs, truth_boxes, truth_labels):
        """
            inputs: [6, 1, 64, 64, 64]
            use origin img/down_4 as another cls feature map
        """

        # feat_4:[batch_size, channels:64, D:16, H:16, W:16]
        features, feat_4 = data_parallel(self.feature_net, inputs)
        fs = features[-1]  # [12, 128, 32, 32, 32]

        self.rpn_logits_flat, self.rpn_deltas_flat = data_parallel(
            self.rpn, fs)

        b, D, H, W, _, num_class = self.rpn_logits_flat.shape

        # print('rpn_logit ', self.rpn_logits_flat.shape)
        self.rpn_logits_flat = self.rpn_logits_flat.view(b, -1, 1)
        # print('rpn_delta ', self.rpn_deltas_flat.shape)
        self.rpn_deltas_flat = self.rpn_deltas_flat.view(b, -1, 6)

        feature_size = fs.shape[2:]       

        if self.feature_size == None or self.feature_size != feature_size :     
            self.rpn_window = make_rpn_windows(fs, self.cfg)
            self.feature_size = feature_size

        self.rpn_proposals = []

        if self.use_rcnn or self.mode in ['eval', 'test']:
            self.rpn_proposals = rpn_nms(self.cfg, self.mode, inputs, self.rpn_window,
                                         self.rpn_logits_flat, self.rpn_deltas_flat)

        if self.mode in ['train', 'valid']:

            self.rpn_labels, self.rpn_label_assigns, self.rpn_label_weights, self.rpn_targets, self.rpn_target_weights = \
                make_rpn_target(self.cfg, self.mode, inputs,
                                self.rpn_window, truth_boxes, truth_labels)

            if self.use_rcnn:
                self.rpn_proposals, self.rcnn_labels, self.rcnn_assigns, self.rcnn_targets = \
                    make_rcnn_target(self.cfg, self.mode, inputs, self.rpn_proposals,
                                     truth_boxes, truth_labels)

        # rcnn proposals
        self.detections = copy.deepcopy(self.rpn_proposals)
        self.ensemble_proposals = copy.deepcopy(self.rpn_proposals)

        if self.use_rcnn:
            if len(self.rpn_proposals) > 0:
                # rcnn on down_4
                rcnn_crops = self.rcnn_crop(feat_4, inputs, self.rpn_proposals)
                # # rcnn on the last feature map
                # last_rcnn_crops = self.rcnn_crop(features[1], inputs, self.rpn_proposals)

                # mixup_crops = 0.5 * rcnn_crops + 0.5 * last_rcnn_crops

                self.rcnn_logits, self.rcnn_deltas = data_parallel(
                    self.rcnn_head, rcnn_crops)
                self.detections, self.keeps = rcnn_nms(self.cfg, self.mode, inputs, self.rpn_proposals,
                                                       self.rcnn_logits, self.rcnn_deltas)

            if self.mode in ['eval']:
                # Ensemble
                fpr_res = get_probability(self.cfg, self.mode, inputs, self.rpn_proposals, self.rcnn_logits,
                                          self.rcnn_deltas)
                self.ensemble_proposals[:, 1] = self.ensemble_proposals[:,
                                                                        1] * 0.5 + fpr_res[:, 0] * 0.5

    def loss(self):

        self.rcnn_cls_loss, self.rcnn_reg_loss, self.iou_loss = \
            torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda()

        self.rpn_cls_loss, self.rpn_reg_loss = rpn_loss(self.rpn_logits_flat, self.rpn_deltas_flat, self.rpn_labels,
                                                        self.rpn_label_weights, self.rpn_targets,
                                                        self.rpn_target_weights, self.cfg, mode=self.mode)

        if self.use_rcnn:
            self.rcnn_cls_loss, self.rcnn_reg_loss = rcnn_loss(self.rcnn_logits, self.rcnn_deltas, self.rcnn_labels,
                                                               self.rcnn_targets)

        self.total_loss = self.rpn_cls_loss + self.rpn_reg_loss \
            + self.rcnn_cls_loss + self.rcnn_reg_loss

        return self.total_loss

    def set_mode(self, mode):
        assert mode in ['train', 'valid', 'eval', 'test']
        self.mode = mode
        if mode in ['train']:
            self.train()
        else:
            self.eval()

    def set_anchor_params(self, anchor_ids, anchor_params):
        self.anchor_ids = anchor_ids
        self.anchor_params = anchor_params

    def crf(self, detections):
        """
            detections: numpy array of detection results [b, z, y, x, d, h, w, p]
        """
        res = []
        config = self.cfg
        anchor_ids = self.anchor_ids
        anchor_params = self.anchor_params
        anchor_centers = []

        for a in anchor_ids:
            # category starts from 1 with 0 denoting background
            # id starts from 0
            cat = a + 1
            dets = detections[detections[:, -1] == cat]
            if len(dets):
                b, p, z, y, x, d, h, w, _ = dets[0]
                anchor_centers.append([z, y, x])
                res.append(dets[0])
            else:
                # Does not have anchor box
                return detections

        pred_cats = np.unique(detections[:, -1]).astype(np.uint8)
        for cat in pred_cats:
            if cat - 1 not in anchor_ids:
                cat = int(cat)
                preds = detections[detections[:, -1] == cat]
                score = np.zeros((len(preds),))
                roi_name = config['roi_names'][cat - 1]

                for k, params in enumerate(anchor_params):
                    param = params[roi_name]
                    for i, det in enumerate(preds):
                        b, p, z, y, x, d, h, w, _ = det
                        d = np.array([z, y, x]) - np.array(anchor_centers[k])
                        prob = norm.pdf(d, param[0], param[1])
                        prob = np.log(prob)
                        prob = np.sum(prob)
                        score[i] += prob

                res.append(preds[score == score.max()][0])

        res = np.array(res)
        return res


def build_model(config):
    return MainNet(config)

if __name__ == '__main__':
    net = MainNet(config)

    input = torch.rand([4, 1, 128, 128, 128])
    input = Variable(input)
