import csv
import os
import numpy as np
from evaluationScript.tools import csvTools
import nrrd
import matplotlib.pyplot as plt
import sys
from config import data_config, train_config


def generate_test_anno(anno_dir, test_list_dir, test_anno_dir):
    """
    generate the annotations csv file for the image which need to be visualized.
    anno_dir: all annotations in one file
    test_list_dir: ct filename list
    test_anno_dir: output to a csv file
    """
    test_list = csvTools.readCSV(test_list_dir)
    anno_list = csvTools.readCSV(anno_dir)

    test_id_list = []

    for id in test_list:
        test_id_list.append(id[0])

    try:
        with open(test_anno_dir, 'w') as csvfile:
            writer = csv.writer(csvfile)
            for anno in anno_list:
                if anno[0] in test_id_list:
                    writer.writerow([anno[0], anno[1], anno[2], anno[3], anno[4]])

    except:
        print("Unexpected error:", sys.exc_info()[0])


def f(data):
    return float(data)


def draw_nms(predicts_list, threshold):
    pd_list = np.array(predicts_list, dtype=np.float32)

    x1 = pd_list[:, 0] - pd_list[:, 3]
    y1 = pd_list[:, 1] - pd_list[:, 3]
    z1 = pd_list[:, 2] - pd_list[:, 3]
    x2 = pd_list[:, 0] + pd_list[:, 3]
    y2 = pd_list[:, 1] + pd_list[:, 3]
    z2 = pd_list[:, 2] + pd_list[:, 3]
    scores = pd_list[:, 4]

    order = scores.argsort()[::-1]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1) * (z2 - z1 + 1)
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        zz1 = np.maximum(z1[i], z1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        zz2 = np.maximum(z2[i], z2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1) * np.maximum(0.0, zz2 - zz1 + 1)
        iou_3d = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou_3d <= threshold)[0]
        order = order[inds + 1]
    bbox = pd_list[keep]

    return bbox.tolist()


def draw_boxes(filename, pid, gt_list, pred_list, outpath):
    # print(filename)
    arr, options = nrrd.read(filename)
    png_dir = outpath + pid
    # print(png_dir)
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    txt_color = '#000000'
    pred_color = '#FFFFFF'
    gt_color = '#DC143C'

    for i, slice in enumerate(arr):
        plt.figure()
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.imshow(slice, cmap="bone")
        # draw prediction
        for axis in pred_list:

            start = int(axis[2] - int(axis[3] / 2))
            end = int(axis[2] + int(axis[3] / 2))
            # print("p_start:{}  p_end:{}".format(start, end))
            if start <= i <= end:
                rect = plt.Rectangle(
                    (axis[0] - axis[3] / 2, axis[1] - axis[3] / 2),
                    axis[3], axis[3],
                    fill=False,
                    edgecolor=pred_color,
                    linewidth=2
                )
                plt.gca().add_patch(rect)
                plt.text(
                    axis[0] - axis[3] / 2, axis[1] - axis[3] / 2,
                    round(data[4], 2),
                    color=txt_color,
                    bbox={'edgecolor': pred_color, 'facecolor': pred_color, 'alpha': 0.5, 'pad': 0}
                )

        # draw ground-truth
        for data in gt_list:
            start = int(data[2] - int(data[3] / 2))
            end = int(data[2] + int(data[3] / 2))
            if start <= i <= end:
                rect = plt.Rectangle(
                    (data[0] - data[3] / 2, data[1] - data[3] / 2),
                    data[3], data[3],
                    fill=False,
                    edgecolor=gt_color,
                    linewidth=2
                )
                plt.gca().add_patch(rect)

        plt.savefig(png_dir + "/{}.png".format(i))
        plt.close()


def draw_one_fold(n):
    # new annos after preprocess
    anno_dir = data_config['new_annos_dir']
    # generate gt data via detection result
    test_anno_dir = 'annotations/test_anno.csv'
    # original img folder
    preprocessed_path = data_config['preprocessed_data_dir']
    # ct need to be visualized
    val_path = "detection/example.csv"
    out_path = "detection/"
    # detection result data 
    result_path = 'results/transformer_conv_fpr/{}_fold/res/100/FROC/submission_ensemble.csv'.format(n)

    generate_test_anno(anno_dir, val_path, test_anno_dir)

    pid_list = []
    pid_data = csvTools.readCSV(val_path)
    for i in pid_data:
        pid_list.append(i[0])

    gt_data = csvTools.readCSV(test_anno_dir)
    pred_data = csvTools.readCSV(result_path)[1:]

    for pid in pid_list:
        gt_list = []
        for i in gt_data:
            if pid == i[0]:
                data = i[1:]
                gt_list.append(data)

        for i in range(len(gt_list)):
            for j in range(4):
                gt_list[i][j] = f(gt_list[i][j])

        pred_list = []
        for i in pred_data:
            if pid == i[0]:
                data = i[1:]
                pred_list.append(data)

        for i in range(len(pred_list)):
            for j in range(5):
                pred_list[i][j] = f(pred_list[i][j])

        pd_list = draw_nms(pred_list, 0.1)
        filename = preprocessed_path + '/' + pid + ".nrrd"

        draw_boxes(filename, pid, gt_list, pd_list, out_path)

        print('-- Finished # {}'.format(filename))


if __name__ == "__main__":
    draw_one_fold(9)
