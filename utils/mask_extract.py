import sys
sys.path.append('../')
import numpy as np
import scipy.ndimage
from skimage import measure, morphology
import SimpleITK as sitk
from multiprocessing import Pool
import os
import nrrd
from config import config
import pandas as pd


def load_itk_image(filename):
    """
    Return img array and [z,y,x]-ordered origin and spacing
    """

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing


def HU2uint8(image, HU_min=-1200.0, HU_max=600.0, HU_nan=-2000.0):
    """
    Convert HU unit into uint8 values. First bound HU values by predfined min
    and max, and then normalize
    image: 3D numpy array of raw HU values from CT series in [z, y, x] order.
    HU_min: float, min HU value.
    HU_max: float, max HU value.
    HU_nan: float, value for nan in the raw CT image.
    """
    image_new = np.array(image)
    image_new[np.isnan(image_new)] = HU_nan

    # normalize to [0, 1]
    image_new = (image_new - HU_min) / (HU_max - HU_min)
    image_new = np.clip(image_new, 0, 1)
    image_new = (image_new * 255).astype('uint8')

    return image_new


def convex_hull_dilate(binary_mask, dilate_factor=1.5, iterations=10):
   
    binary_mask_dilated = np.array(binary_mask)
   
    struct = scipy.ndimage.morphology.generate_binary_structure(3, 1)
    binary_mask_dilated = scipy.ndimage.morphology.binary_dilation(
        binary_mask_dilated, structure=struct, iterations=10)

    return binary_mask_dilated



def apply_mask(image, binary_mask, pad_value=170,
               bone_thred=210, remove_bone=False):
   
    binary_mask_dilated = convex_hull_dilate(binary_mask)

    binary_mask_extra = binary_mask_dilated ^ binary_mask

    # replace image values outside binary_mask_dilated as pad value
    image_new = image * binary_mask_dilated + \
                pad_value * (1 - binary_mask_dilated).astype('uint8')

    # set bones in extra mask to 170 (ie convert HU > 482 to HU 0;
    # water).
    if remove_bone:
        image_new[image_new * binary_mask_extra > bone_thred] = pad_value

    return image_new


def resample(image, spacing, new_spacing=[1.0, 1.0, 1.0], order=1):
   
    # shape can only be int, so has to be rounded.
    new_shape = np.round(image.shape * spacing / new_spacing)

    # the actual spacing to resample.
    resample_spacing = spacing * image.shape / new_shape

    resize_factor = new_shape / image.shape

    image_new = scipy.ndimage.interpolation.zoom(image, resize_factor,
                                                 mode='nearest', order=order)

    return (image_new, resample_spacing)


def get_lung_box(binary_mask, new_shape, margin=5):
    
    # list of z, y x indexes that are true in binary_mask
    z_true, y_true, x_true = np.where(binary_mask)
    old_shape = binary_mask.shape

    lung_box = np.array([[np.min(z_true), np.max(z_true)],
                         [np.min(y_true), np.max(y_true)],
                         [np.min(x_true), np.max(x_true)]])
    lung_box = lung_box * 1.0 * \
               np.expand_dims(new_shape, 1) / np.expand_dims(old_shape, 1)
    lung_box = np.floor(lung_box).astype('int')

    z_min, z_max = lung_box[0]
    y_min, y_max = lung_box[1]
    x_min, x_max = lung_box[2]

    # extend the lung_box by a margin
    lung_box[0] = max(0, z_min - margin), min(new_shape[0], z_max + margin)
    lung_box[1] = max(0, y_min - margin), min(new_shape[1], y_max + margin)
    lung_box[2] = max(0, x_min - margin), min(new_shape[2], x_max + margin)

    return lung_box


def preprocess(params):
    pid, lung_mask_dir, img_dir, save_dir, do_resample = params

    print(f'Preprocessing {pid}...')

    binary_mask, _, _ = load_itk_image(os.path.join(lung_mask_dir, '%s.mhd' % (pid)))
    img, origin, spacing = load_itk_image(os.path.join(img_dir, '%s.mhd' % (pid)))

    img = HU2uint8(img)
    seg_img = apply_mask(img, binary_mask)

    if do_resample:
        print('Resampling...')
        seg_img, resampled_spacing = resample(seg_img, spacing, order=3)

    lung_box = get_lung_box(binary_mask, seg_img.shape)

    z_min, z_max = lung_box[0]
    y_min, y_max = lung_box[1]
    x_min, x_max = lung_box[2]

    seg_img = seg_img[z_min:z_max, y_min:y_max, x_min:x_max]

    nrrd.write(os.path.join(save_dir, '%s_seg.nrrd' % (pid)), seg_img)


def main():
    do_resample = True
    lung_mask_dir = config['lung_mask_dir']
    
    seriesuids_dir = '../split/9_val.csv'

    img_dir = os.path.join(config['data_dir'], 'subset9')
    save_dir = os.path.join(config['preprocessed_data_dir'])
    print('save dir ', save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    uids = pd.read_csv(seriesuids_dir, header=None)[0]

    params_lists = []
    for pid in uids:
        params_lists.append([pid, lung_mask_dir, img_dir, save_dir, do_resample])

    pool = Pool(processes=10)
    pool.map(preprocess, params_lists)

    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
