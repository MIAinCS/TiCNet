import os

import numpy as np
from torch.autograd import Variable

from net.layer.losses import *


def rcnn_loss(logits, deltas, labels, targets):
    batch_size, num_class = logits.size(0), logits.size(1)

    # Weighted cross entropy for imbalance class distribution
    weight = torch.ones(num_class).cuda()
    total = len(labels)
    for i in range(num_class):
        num_pos = float((labels == i).sum())
        num_pos = max(num_pos, 1)
        weight[i] = total / num_pos

    weight = weight / weight.sum()
    rcnn_cls_loss = F.cross_entropy(logits, labels, weight=weight, size_average=True)

    # If multi-class classification, compute the confusion metric to understand the mistakes
    confusion_matrix = np.zeros((num_class, num_class))
    probs = F.softmax(logits, dim=1)
    v, cat = torch.max(probs, dim=1)
    for i in labels.nonzero():
        i = i.item()
        confusion_matrix[labels.long().detach()[i].item()][cat[i].detach().item()] += 1

    num_pos = len(labels.nonzero())

    if num_pos > 0:
        # one hot encode
        select = Variable(torch.zeros((batch_size, num_class))).cuda()
        select.scatter_(1, labels.view(-1, 1), 1)
        select[:, 0] = 0
        select = select.view(batch_size, num_class, 1).expand((batch_size, num_class, 6)).contiguous().bool()

        deltas = deltas.view(batch_size, num_class, 6)
        deltas = deltas[select].view(-1, 6)

        rcnn_reg_loss = 0

        for i in range(6):
            l = F.smooth_l1_loss(deltas[:, i], targets[:, i])
            rcnn_reg_loss += l
        


    else:
        rcnn_reg_loss = Variable(torch.cuda.FloatTensor(1).zero_()).sum()

    return rcnn_cls_loss, rcnn_reg_loss




def rcnn_iou_loss(rpn_proposals, truth_box):
    iou_loss = 0
    
    for slice_idx in range(len(truth_box)):
        
        slice_rpn = rpn_proposals[rpn_proposals[:,0] == slice_idx, 2:-1]
        for nodule_rpn in slice_rpn: 
               
            for nodule_box in truth_box[slice_idx]:
                if box_ciou(nodule_rpn, torch.from_numpy(nodule_box).cuda()) < 0:
                    print("Unavailable box.")
                iou_score = box_ciou(nodule_rpn, torch.from_numpy(nodule_box).cuda())
                iou_loss += iou_score
                
    return iou_loss

