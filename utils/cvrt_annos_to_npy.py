import sys

sys.path.append('../')
from util import *
import csv
import config as config


def load_itk_image(filename):
    """
    Return img array and [z,y,x]-ordered origin and spacing
    """

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing


def get_ct_data(ct_file):
    seriesuid = ct_file.split('/')[-1][:-4]
    img, Origin, Spacing = load_itk_image(ct_file)

    return seriesuid, Origin, Spacing


def get_anno_dict(anno_file):
    anno_data = pd.read_csv(anno_file)

    uid = anno_data["seriesuid"]
    data = anno_data[["coordX", "coordY", "coordZ", "diameter_mm"]]
    data = np.array(data)
    anno_dict = dict([(id, []) for id in uid])
    for i in range(len(uid)):
        anno_dict[uid[i]].append(data[i])

    return anno_dict


def generate_label(annos_dict, seriesuids_dir, img_dir):
    all_uids = pd.read_csv(seriesuids_dir, header=None)[0]

    for uid in all_uids:
        origin = np.load(img_dir + '/' + uid + '_origin.npy')
        spacing = np.load(img_dir + '/' + uid + '_spacing.npy')
        ebox = np.load(img_dir + '/' + uid + '_ebox.npy')
        new_annos = []
        if uid in annos_dict.keys():
            annos = annos_dict[uid]
            for anno in annos:
                anno[[0, 1, 2]] = anno[[2, 1, 0]]
                coord = anno[:-1]
                new_coord = worldToVoxelCoord(coord, origin, spacing) * spacing - ebox
                new_coord = np.append(new_coord, anno[-1])
                new_annos.append(new_coord)
            annos_dict[uid] = new_annos
        else:
            print(f'{uid} does not have any nodules.')
        np.save(os.path.join(img_dir, '%s_bboxes.npy' % (uid)), np.array(new_annos))
        print(f'Finished masks to bboxes {uid}')


def annotation_to_npy(annos_dir, seriesuids_dir, img_dir, annos_save_dir):
    annos_dict = get_anno_dict(annos_dir)
    generate_label(annos_dict, seriesuids_dir, img_dir)

    new_annos_path = config['new_annos_dir']
    try:
        with open(new_annos_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"])

            for uid in annos_dict.keys():
                for annos in annos_dict[uid]:
                    writer.writerow([uid, annos[2], annos[1], annos[0], annos[3]])

    except:
        print("Unexpected error:", sys.exc_info()[0])


def annotation_exclude_to_npy(annos_excluded_dir, img_dir, annos_save_dir):
    try:
        annos_exclude_dict = get_anno_dict(annos_excluded_dir)
    except:
        print("Unexpected error:", sys.exc_info()[0])

    try:
        for uid in annos_exclude_dict.keys():
            annos = annos_exclude_dict[uid]
            origin = np.load(img_dir + '/' + uid + '_origin.npy')
            spacing = np.load(img_dir + '/' + uid + '_spacing.npy')
            ebox = np.load(img_dir + '/' + uid + '_ebox.npy')
            new_annos_exclude = []
            for anno in annos:
                anno[[0, 1, 2]] = anno[[2, 1, 0]]
                coord = anno[:-1]
                new_coord = worldToVoxelCoord(coord, origin, spacing) * spacing - ebox
                new_coord = np.append(new_coord, anno[-1])
                new_annos_exclude.append(new_coord)
            annos_exclude_dict[uid] = new_annos_exclude
    except:
        print("Unexpected error:", sys.exc_info()[0])

    new_annos_exclude_path = config['new_annos_excluded_dir']
    try:
        with open(new_annos_exclude_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"])

            for uid in annos_exclude_dict.keys():
                for annos in annos_exclude_dict[uid]:
                    writer.writerow([uid, annos[2], annos[1], annos[0], annos[3]])
    except:
        print("Unexpected error:", sys.exc_info()[0])


if __name__ == '__main__':
    annos_dir = config['annos_dir']
    annos_excluded_dir = config['annos_excluded_dir']
    seriesuids_dir = config['seriesuids_dir']
    img_dir = config['preprocessed_data_dir']
    annos_save_dir = config['annos_save_dir']

    os.makedirs(annos_save_dir, exist_ok=True)

    annotation_to_npy(annos_dir, seriesuids_dir, img_dir, annos_save_dir)
    # annotation_excluded_to_npy(annos_excluded_dir, img_dir, annos_save_dir)
