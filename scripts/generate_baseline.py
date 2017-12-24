#!/usr/bin/env python
"""
generate_baselines.py creates tsv files containing features for training 
the 'ResNet' baseline from the paper.
"""

import base64
import numpy as np
import cv2
import csv
import json
import os
import caffe
import sys
from scipy.ndimage import zoom
import random
random.seed(1)

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
KARPATHY_SPLITS = './data/coco_splits/karpathy_%s_images.txt' # train,val,test

PROTOTXT='baseline/ResNet-101-deploy.prototxt'
WEIGHTS='baseline/ResNet-101-model.caffemodel'

IMAGE_DIR = 'data/images/'

ATT_WIDTH = 10.0
ATT_HEIGHT = 10.0


def load_karpathy_splits(dataset='train'):
  imgIds = set()
  with open(KARPATHY_SPLITS % dataset) as data_file:
    for line in data_file:
      imgIds.add(int(line.split()[-1]))
  return imgIds


def load_image_ids():
  ''' Map image ids to file paths. '''
  id_to_path = {}
  for fname in ['image_info_test2014.json', 'captions_val2014.json', 'captions_train2014.json']:
    with open('data/coco/%s' % fname) as f:
      data = json.load(f)
      for item in data['images']:
        image_id = int(item['id'])
        filepath = item['file_name'].split('_')[1] + '/' + item['file_name']
        id_to_path[image_id] = filepath
  print 'Loaded %d image ids' % len(id_to_path)
  return id_to_path


def get_detections_from_im(net, image_id, im_file):

  im = cv2.imread(IMAGE_DIR+im_file) # shape (rows, columns, channels)
  im_height = im.shape[0]
  im_width = im.shape[1]
  num_boxes = int(ATT_HEIGHT*ATT_WIDTH)
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= np.array([[[103.1, 115.9, 123.2]]]) # BGR pixel mean
  blob = np.zeros((1, im.shape[0], im.shape[1], 3), dtype=np.float32)
  blob[0, :, :, :] = im_orig
  blob = blob.transpose((0, 3, 1, 2))
  net.blobs['data'].reshape(*(blob.shape))
  forward_kwargs = {'data': blob.astype(np.float32, copy=False)}
  blobs_out = net.forward(**forward_kwargs)

  # Bilinear pooling to 2048 x 10 x 10 output for ResNet features
  res5c = net.blobs['res5c'].data[0]
  zoom_factor = (1,ATT_HEIGHT/res5c.shape[-2],ATT_WIDTH/res5c.shape[-1])
  adaptive_res5c = zoom(res5c, zoom_factor, order=1)
  resnet_features = adaptive_res5c.reshape((2048, int(ATT_HEIGHT*ATT_WIDTH))).T.copy()
  ResNetBaseline = {
      'image_id': image_id,
      'image_h': im_height,
      'image_w': im_width,
      'num_boxes': num_boxes,
      'boxes': base64.b64encode(np.zeros((num_boxes,0), dtype=np.float32)),
      'features': base64.b64encode(resnet_features)
  }
  return ResNetBaseline


if __name__ == "__main__":

  tsv_files = ['karpathy_train_resnet101_baseline.tsv.0',
          'karpathy_train_resnet101_baseline.tsv.1',
          'karpathy_val_resnet101_baseline.tsv',
          'karpathy_test_resnet101_baseline.tsv']
  train = list(load_karpathy_splits(dataset='train'))
  random.shuffle(train)
  image_id_sets = [set(train[:len(train)/2]),
               set(train[len(train)/2:]),
               load_karpathy_splits(dataset='val'),
               load_karpathy_splits(dataset='test')]
               
  if not os.path.exists('data/tsv/'):
    os.makedirs('data/tsv/')
  if not os.path.exists('data/tsv/trainval/'):
    os.makedirs('data/tsv/trainval/')
    
  id_to_path = load_image_ids()
  caffe.set_mode_gpu()
  caffe.set_device(0)
  net = caffe.Net(PROTOTXT, caffe.TEST, weights=WEIGHTS)
    
  for tsv,image_ids in zip(tsv_files, image_id_sets):
    out_file = 'data/tsv/trainval/' + tsv
    with open(out_file, 'wb') as resnet_tsv_out:
      print 'Writing to %s' % out_file
      resnet_writer = csv.DictWriter(resnet_tsv_out, delimiter = '\t', fieldnames = FIELDNAMES)
      count = 0
      for image_id in image_ids:
        resnet_baseline = get_detections_from_im(net, image_id, id_to_path[image_id])
        resnet_writer.writerow(resnet_baseline)
        count += 1
        if count % 1000 == 0:
          print '%d / %d' % (count, len(image_ids))


