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

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

PROTOTXT='baselines/ResNet-101-deploy.prototxt'
WEIGHTS='baselines/ResNet-101-model.caffemodel'

IMAGE_DIR = '/data/coco/images/'

ATT_WIDTH = 10.0
ATT_HEIGHT = 10.0


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


def get_detections_from_im(net, image_id, im_file, boxes):

  im = cv2.imread(IMAGE_DIR+im_file) # shape (rows, columns, channels)
  im_height = im.shape[0]
  im_width = im.shape[1]
  num_boxes = boxes.shape[0]
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
  #zoom_factor = (1,ATT_HEIGHT/res5c.shape[-2],ATT_WIDTH/res5c.shape[-1])
  #adaptive_res5c = zoom(res5c, zoom_factor, order=1)
  #resnet_features = adaptive_res5c.reshape((2048, int(ATT_HEIGHT*ATT_WIDTH))).T.copy()
  #ResNetBaseline = {
  #    'image_id': image_id,
  #    'image_h': im_height,
  #    'image_w': im_width,
  #    'num_boxes': int(ATT_HEIGHT*ATT_WIDTH),
  #    'boxes': base64.b64encode(np.zeros((int(ATT_HEIGHT*ATT_WIDTH),0), dtype=np.float32)),
  #    'features': base64.b64encode(resnet_features)
  #}

  # Convert bbox to feature coordinates and mean pool for Bbox features
  bbox_features = np.zeros((num_boxes,2048), dtype=np.float32)
  feat_width = res5c.shape[-1]
  feat_height = res5c.shape[-2]
  for i,box in enumerate(boxes): # rectangle (x1, y1, x2, y2)
    # Convert box to feature space
    x1 = box[0]/im_width*feat_width
    x2 = box[2]/im_width*feat_width
    y1 = box[1]/im_height*feat_height
    y2 = box[3]/im_height*feat_height
    # For each spatial position, calculate the enclosed proportion
    x_enc = np.array([max(0,min(x2,j+1)-max(x1,j)) for j in range(feat_width)])
    y_enc = np.array([max(0,min(y2,j+1)-max(y1,j)) for j in range(feat_height)])
    p = np.outer(y_enc,x_enc)
    bbox_features[i, :] = np.sum((p*res5c).reshape(2048,-1),axis=1)/p.sum()
  BboxBaseline = {
      'image_id': image_id,
      'image_h': im_height,
      'image_w': im_width,
      'num_boxes': num_boxes,
      'boxes': base64.b64encode(boxes),
      'features': base64.b64encode(bbox_features)
  }
  return BboxBaseline
  #return (ResNetBaseline, BboxBaseline)



def generate_baselines(gpu_id, infile, resnet_outfile, bbox_outfile):

  id_to_path = load_image_ids()
  caffe.set_mode_gpu()
  caffe.set_device(gpu_id)
  net = caffe.Net(PROTOTXT, caffe.TEST, weights=WEIGHTS)

  with open(infile) as tsvfile:
    reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames = FIELDNAMES)
    with open(resnet_outfile, 'wb') as resnet_tsv_out:
      resnet_writer = csv.DictWriter(resnet_tsv_out, delimiter = '\t', fieldnames = FIELDNAMES)
      with open(bbox_outfile, 'wb') as bbox_tsv_out:
        bbox_writer = csv.DictWriter(bbox_tsv_out, delimiter = '\t', fieldnames = FIELDNAMES)
        for item in reader:
          image_id = int(item['image_id'])
          num_boxes = int(item['num_boxes'])
          boxes = np.frombuffer(base64.decodestring(item['boxes']), 
                      dtype=np.float32).reshape((num_boxes,4))
          bbox_baseline = get_detections_from_im(net, image_id, id_to_path[image_id], boxes)
          #resnet_baseline, bbox_baseline = get_detections_from_im(net, image_id, id_to_path[image_id], boxes)
          #resnet_writer.writerow(resnet_baseline)
          bbox_writer.writerow(bbox_baseline)

if __name__ == "__main__":

  # TODO start training, then extract full size (delete first)
  gpu_id = 1
  data_dir = '/data/coco/tsv/'
  for tsv in [
#      'karpathy_test_resnet101_faster_rcnn_final_test.tsv',
#      'karpathy_train_resnet101_faster_rcnn_final_test.tsv.0',
      'karpathy_train_resnet101_faster_rcnn_final_test.tsv.1',
      'karpathy_val_resnet101_faster_rcnn_final_test.tsv'\
      ]:
    generate_baselines(gpu_id, data_dir+tsv, 
      data_dir+tsv.replace('faster_rcnn','baseline_resnet_delete2'), 
      data_dir+tsv.replace('faster_rcnn','baseline_bbox_fixed'))


