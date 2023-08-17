from net.layer.losses import *


def rpn_loss(logits, deltas, labels, label_weights, targets, target_weights, cfg, mode='train', delta_sigma=3.0):
    batch_size, num_windows, num_classes = logits.size()
    batch_size_k = batch_size
    labels = labels.long()

    # Calculate classification score
    pos_correct, pos_total, neg_correct, neg_total = 0, 0, 0, 0
    batch_size = batch_size * num_windows
    logits = logits.view(batch_size, num_classes)
    labels = labels.view(batch_size, 1)
    label_weights = label_weights.view(batch_size, 1)

    # Make sure OHEM is performed only in training mode
    if mode in ['train']:
        num_hard = cfg['num_hard']
    else:
        num_hard = 10000000

    rpn_cls_loss, pos_correct, pos_total, neg_correct, neg_total = \
        binary_cross_entropy_with_hard_negative_mining(logits, labels, label_weights, batch_size_k, num_hard)

    # rpn_cls_loss, pos_correct, pos_total, neg_correct, neg_total = \
    #     focal_loss(logits, labels, label_weights)

    # Calculate regression
    deltas = deltas.view(batch_size, 6)
    targets = targets.view(batch_size, 6)

    index = (labels != 0).nonzero()[:, 0]
    deltas = deltas[index]
    targets = targets[index]

    rpn_reg_loss = 0

    for i in range(6):
        l = F.smooth_l1_loss(deltas[:, i], targets[:, i])
        rpn_reg_loss += l

    return rpn_cls_loss, rpn_reg_loss
