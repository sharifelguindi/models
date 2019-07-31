from __future__ import print_function
import numpy as np
import os, fnmatch
import h5py
import cv2
from scipy import ndimage
import PIL
from skimage import measure
import SimpleITK as sitk

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def find_file(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

def bbox2_3D(img, pad):

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin - pad, rmax + pad, cmin - pad, cmax + pad, zmin - pad, zmax + pad

def normalize_equalize_smooth_MR(arr, clahe):

    length, width, height = np.shape(arr)
    norm_arr = np.zeros(np.shape(arr), dtype='uint16')
    norm_arr = cv2.normalize(arr, norm_arr, 0, 65535, cv2.NORM_MINMAX)
    norm_arr = np.clip(norm_arr, 0, 65535)
    norm_eq = np.zeros((length, 3, width, height), dtype='uint16')
    for ii in range(0, length):
        if ii == 0:
            img_0 = clahe.apply(norm_arr[ii+0, :, :].astype('uint16'))
            img_1 = clahe.apply(norm_arr[ii+0, :, :].astype('uint16'))
            img_2 = clahe.apply(norm_arr[ii+1, :, :].astype('uint16'))
        elif ii == length-1:
            img_0 = clahe.apply(norm_arr[ii-1, :, :].astype('uint16'))
            img_1 = clahe.apply(norm_arr[ii, :, :].astype('uint16'))
            img_2 = clahe.apply(norm_arr[ii, :, :].astype('uint16'))
        else:
            img_0 = clahe.apply(norm_arr[ii-1, :, :].astype('uint16'))
            img_1 = clahe.apply(norm_arr[ii+0, :, :].astype('uint16'))
            img_2 = clahe.apply(norm_arr[ii+1, :, :].astype('uint16'))

        norm_eq[ii, 0, :, :] = smooth_image(img_0).astype('uint16')
        norm_eq[ii, 1, :, :] = 255 - smooth_image(img_1).astype('uint16')
        norm_eq[ii, 2, :, :] = smooth_image(img_2).astype('uint16')

    norm_arr_ds = np.zeros(np.shape(norm_eq), dtype='uint8')
    norm_arr_ds = cv2.normalize(norm_eq, norm_arr_ds, 0, 255, cv2.NORM_MINMAX)
    return norm_arr_ds.astype('uint8')

def smooth_image(arr, t_step=0.125, n_iter=5):
    img = sitk.GetImageFromArray(arr)
    img = sitk.CurvatureFlow(image1=img,
                             timeStep=t_step,
                             numberOfIterations=n_iter)
    arr_smoothed = sitk.GetArrayFromImage(img)
    return arr_smoothed

def data_export_MR_3D(scan, mask, save_path, p_num, struct_name):

    ## Parameters for dataset
    length, height, width = np.shape(scan)
    max_class = np.max(mask)
    ## Create folders to store images/masks
    save_path = os.path.join(save_path, struct_name, 'processed')
    if not os.path.exists(os.path.join(save_path,'PNGImages')):
        os.makedirs(os.path.join(save_path,'PNGImages'))
    if not os.path.exists(os.path.join(save_path, 'SegmentationClass')):
        os.makedirs(os.path.join(save_path, 'SegmentationClass'))
    if not os.path.exists(os.path.join(save_path, 'SegmentationVis')):
        os.makedirs(os.path.join(save_path, 'SegmentationVis'))

    ## Verify size of scan data and mask data equivalent
    if scan.shape == mask.shape:

        clahe = cv2.createCLAHE(clipLimit=20, tileGridSize=(int(scan.shape[1] / 16), int(scan.shape[2] / 16)))
        scan_norm = normalize_equalize_smooth_MR(scan, clahe)
        mask_norm = mask.transpose(2, 1, 0)
        msk = np.zeros((height, width), dtype='uint8')
        msk_vis = np.zeros((height, width), dtype='uint8')
        rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3D(mask, 10)

        for i in range(0, length):
            img = scan_norm[i, :, :, :].transpose(2, 1, 0)
            msk[:, :] = mask_norm[:, :, i]
            msk_vis[:, :] = cv2.normalize(mask_norm[:, :, i], msk_vis, 0, 255, cv2.NORM_MINMAX)
            unique, counts = np.unique(msk, return_counts=True)
            vals = dict(zip(unique, counts))
            if len(vals) > 0:
                img_name = os.path.join('PNGImages','ax' + str(p_num) + '_' + str(i))
                gt_name = os.path.join('SegmentationClass','ax' + str(p_num) + '_' + str(i))
                vis_name = os.path.join('SegmentationVis','ax' + str(p_num) + '_' + str(i))

                PIL.Image.fromarray(img).save(os.path.join(save_path, img_name + '.png'))
                PIL.Image.fromarray(msk).save(os.path.join(save_path, gt_name + '.png'))
                PIL.Image.fromarray(msk_vis).save(os.path.join(save_path, vis_name + '.png'))

