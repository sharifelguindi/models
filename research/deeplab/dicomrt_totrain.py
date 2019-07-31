# Written by: Sharif Elguindi, MS, DABR
# ==============================================================================
#
# This script returns PNG image files and associated masks for 2D training of images
# using FCN architectures in tensorflow.
#
# Usage:
#
#   python dicomrt_to_traindata.py \
#   --numShards='number of pieces to break dataset into for transfer purposes
#   --rawDir='path\to\data\'
#   --saveDir='path\to\save\'
#   --datasetName='structure_name_to_search_for'

from __future__ import print_function
import sys
sys.path.append('../')
# Add tensorflow slim package to python path; Download from:
# https://github.com/tensorflow/models/tree/master/research/slim
if "D:\\pythonProjects\\models\\research\\slim" not in sys.path:
    sys.path.append("D:\\pythonProjects\\models\\research\\slim")

import numpy as np
import os
import tensorflow as tf
import h5py
import sys
import glob
import math
import datasets.build_data as build_data
from functions import *
import random

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('numShards', 10,
                     'Split train/val data into chucks if large dateset >2-3000 (default, 1)')

flags.DEFINE_string('rawDir', 'D:\\publicDatasets\\imgSeg\\SHARP2019\\TrainingData\\',
                    'absolute path to where raw data is collected from.')

flags.DEFINE_string('saveDir', 'D:\\trainingTF\\',
                    'absolute path to where processed data is saved.')

flags.DEFINE_string('datasetName', 'prostateAxialT2',
                    'string name of structure to export')

def create_tfrecord(structure_path):

    planeList = ['ax', 'cor', 'sag']
    planeDir = ['Axial', 'Coronal', 'Sag']
    filename_train = 'train_'
    filename_val = 'val_'
    i = 0
    for plane in planeList:

        file_base = os.path.join(structure_path, 'processed', 'ImageSets', planeDir[i])
        if not os.path.exists(file_base):
            os.makedirs(file_base)
        f = open(os.path.join(file_base, filename_train + plane + '.txt'), 'a')
        f.truncate()
        k = 0
        path = os.path.join(structure_path, 'processed', 'PNGImages')
        pattern = plane + '*.png'
        files = find(pattern, path)
        for file in files:
            if file.find(plane) > 0 \
                    and (file.find(plane + '300_') < 1
                    and  file.find(plane + '310_') < 1):
                h = file.split(os.sep)
                f.write(h[-1].replace('.png','') +'\n')
                k = k + 1
        f.close()
        print(filename_train + plane, k)

        if not os.path.exists(file_base):
            os.makedirs(file_base)
        f = open(os.path.join(file_base, filename_val + plane + '.txt'), 'a')
        f.truncate()
        k = 0
        for file in files:
            if file.find(plane) > 0 \
                    and (file.find(plane + '300_') > 0
                    or   file.find(plane + '310_') > 0):
                h = file.split(os.sep)
                f.write(h[-1].replace('.png','') +'\n')
                k = k + 1
        f.close()
        print(filename_val + plane, k)
        i = i + 1

        dataset_splits = glob.glob(os.path.join(file_base, '*.txt'))
        for dataset_split in dataset_splits:
            _convert_dataset(dataset_split, FLAGS.numShards, structure_path, plane)

    return

def _convert_dataset(dataset_split, _NUM_SHARDS, structure_path, plane):
  """Converts the specified dataset split to TFRecord format.

  Args:
    dataset_split: The dataset split (e.g., train, test).

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  image_folder = os.path.join(structure_path, 'processed', 'PNGImages')
  semantic_segmentation_folder = os.path.join(structure_path, 'processed', 'SegmentationClass')
  image_format = label_format = 'png'

  if not os.path.exists(os.path.join(structure_path, 'tfrecord'+ '_' + plane)):
      os.makedirs(os.path.join(structure_path, 'tfrecord'+ '_' + plane))

  dataset = os.path.basename(dataset_split)[:-4]
  sys.stdout.write('Processing ' + dataset)
  filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
  random.shuffle(filenames)
  print(filenames)
  num_images = len(filenames)
  num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

  image_reader = build_data.ImageReader('png', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)

  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(
        structure_path, 'tfrecord'+ '_' + plane,
        '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, len(filenames), shard_id))
        sys.stdout.flush()
        # Read the image.
        image_filename = os.path.join(
            image_folder, filenames[i] + '.' + image_format)
        image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
        height, width = image_reader.read_image_dims(image_data)
        # Read the semantic segmentation annotation.
        seg_filename = os.path.join(
            semantic_segmentation_folder,
            filenames[i] + '.' + label_format)
        seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()
        seg_height, seg_width = label_reader.read_image_dims(seg_data)
        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')
        # Convert to tf example.
        example = build_data.image_seg_to_tfexample(
            image_data, str.encode(filenames[i],'utf-8'), height, width, seg_data)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()

def main(unused_argv):

    data_path = FLAGS.rawDir
    p_num = 1
    incomplete = []
    patient_sets = find('mask_total*', data_path)
    patient_sets.sort()
    for patient in patient_sets:
        s = h5py.File(patient.replace('mask_total', 'scan'), 'r')
        m = h5py.File(patient, 'r')
        # scan, mask should be up shape: (scan length, height, width)
        scan = s['scan'][:]
        mask = m['mask_total'][:]
        unique, counts = np.unique(mask, return_counts=True)
        print('Saving patient dataset: ' + patient)
        data_export_MR_3D(scan, mask, FLAGS.saveDir, p_num, FLAGS.datasetName)
        p_num = p_num + 1
    create_tfrecord(os.path.join(FLAGS.saveDir, FLAGS.datasetName))

if __name__ == '__main__':
  tf.app.run()

