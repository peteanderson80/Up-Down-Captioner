import caffe
import numpy as np
import ast
import base64
import csv
import random
import sys
import traceback
import re
import json
from collections import defaultdict,Counter

csv.field_size_limit(sys.maxsize)
np.random.seed()

class RCNNTestDataLayer(caffe.Layer):
  """
  Data layer that outputs pre-computed faster RCNN detections.
  """

  def _shuffle_roidb_inds(self):
    """ Randomly permute the training data - but not for test. """
    self._cur = 0  
  
  def _get_next_minibatch_inds(self):
    """ Return the data indices for the next minibatch. """
    db_inds = range(self._cur, min(len(self._data),self._cur + self._batch_size))
    self._cur += self._batch_size
    return db_inds        
    
  def _peek_at_dataset(self):
    """ Establish the feature length. """
    with open(self.params['feature_sources'][0]) as tsvfile:
      reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames = self._fieldnames)
      for item in reader:
        item['features'] = np.frombuffer(base64.decodestring(item['features']), 
            dtype=np.float32).reshape((int(item['num_boxes']),-1))
        self._feature_len = item['features'].shape[1]
        break
    
  def load_dataset(self, rank=0, num_gpus=1):
    self._data = []
    for src in self.params['feature_sources'][rank::num_gpus]:
      print 'Loading features from: %s' % src
      with open(src) as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames = self._fieldnames)
        for item in reader:
          try:
            item['image_id'] = int(item['image_id'])
            item['num_boxes'] = int(item['num_boxes'])
            assert item['num_boxes'] > 0
            item['image_h'] = int(item['image_h'])
            assert item['image_h'] > 0
            item['image_w'] = int(item['image_w'])
            assert item['image_w'] > 0       
            # Decode the data immediately
            for field in ['boxes', 'features']:
              item[field] = np.frombuffer(base64.decodestring(item[field]), 
                  dtype=np.float32).reshape((item['num_boxes'],-1))
              item[field] = item[field][:self._max_boxes,:]
            assert item['features'].shape[1] == self._feature_len, item['features'].shape[1]
            item['num_boxes'] = min(item['num_boxes'], self._max_boxes)
            self._data.append(item)
          except Exception:
            print "Skipping image id: %d" % item['image_id']
            print(traceback.format_exc())
    print 'Feature length: %d' % (self._feature_len)
    self._shuffle_roidb_inds()
          
  def _load_params(self):
    self._fieldnames = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
    self.params = ast.literal_eval(self.param_str)
    self._batch_size = int(self.params['batch_size'])
    self._first_box_is_pool = bool(self.params['pool']) # First feature is average pool
    if 'max_boxes' in self.params:
      self._max_boxes = int(self.params['max_boxes'])
    else:
      self._max_boxes = 100     
       
  def setup(self, bottom, top):
    assert len(top) == 4, \
        "Outputs 4 blobs (image_id, num_boxes, boxes, features)."        
    self._load_params()
    self._peek_at_dataset()
    
  def reshape(self, bottom, top):
    top[0].reshape(self._batch_size, 3) # image_id, image_h, image_w
    top[1].reshape(self._batch_size, 1) # num boxes
    if self._first_box_is_pool:
      top[2].reshape(self._batch_size, self._max_boxes+1, 4) # boxes
      top[3].reshape(self._batch_size, self._max_boxes+1, self._feature_len) # features
    else:
      top[2].reshape(self._batch_size, self._max_boxes, 4) # boxes
      top[3].reshape(self._batch_size, self._max_boxes, self._feature_len) # features
  
  def forward(self, bottom, top):
    """ Output image_id, num_boxes, boxes, features. """
    if not hasattr(self, '_cur'):
      self.load_dataset()
    db_inds = self._get_next_minibatch_inds()
    top[2].data[...] = 0
    top[3].data[...] = 0
    for i,ix in enumerate(db_inds):
      item = self._data[ix]
      top[0].data[i] = [item['image_id'], item['image_w'], item['image_h']]
      top[1].data[i] = item['num_boxes']
      if self._first_box_is_pool:
        top[2].data[i,1:item['num_boxes']+1,:] = item['boxes'] 
        top[3].data[i,0,:self._feature_len] = np.mean(item['features'], axis=0)
        top[3].data[i,1:item['num_boxes']+1,:self._feature_len] = item['features']
      else:
        top[2].data[i,:item['num_boxes'],:] = item['boxes']
        top[3].data[i,:item['num_boxes'],:self._feature_len] = item['features']

  def backward(self, top, propagate_down, bottom):
    """This layer does not propagate gradients."""
    pass    


class RCNNCaptionTrainDataLayer(RCNNTestDataLayer):
  """
  Data layer that outputs pre-computed faster RCNN detections and captions
  """
  
  def _shuffle_roidb_inds(self):
    """Randomly permute the training data."""
    self._perm = np.random.permutation(np.arange(len(self._data)))
    self._cur = 0

  def _get_next_minibatch_inds(self):
    """Return the data indices for the next minibatch."""
    if self._cur + self._batch_size >= len(self._data):
        self._shuffle_roidb_inds()
    db_inds = self._perm[self._cur:self._cur + self._batch_size]
    self._cur += self._batch_size
    return db_inds
    
  def load_dataset(self, rank=0, num_gpus=1):
    # Load image features
    super(RCNNCaptionTrainDataLayer, self).load_dataset(rank, num_gpus)
    # Load captions
    id_to_caption = defaultdict(list)
    for src in self.params['caption_sources']:
      print 'Loading captions from: %s' % src
      with open(src) as txtfile:
        for line in txtfile.readlines():
          image_id = int(line.split('.jpg')[0].split('_')[-1])
          seq = [int(w) for w in line.split(' | ')[-1].split()]
          id_to_caption[image_id].append(seq)
      print 'Loaded %d image ids' % len(id_to_caption)
    # Add captions to image features  
    for item in self._data:
      captions = id_to_caption[item['image_id']]
      assert len(captions) >= 5
      item['captions'] = captions
    self._shuffle_roidb_inds()
          
  def setup(self, bottom, top):
    assert len(top) == 6, \
        "Outputs 6 blobs (image_id, input_sentence, \
        target_sentence, num_boxes, boxes, features)."
    self._load_params()
    self._ignore_label = int(self.params['ignore_label'])
    self._end_of_sequence = int(self.params['end_of_sequence'])
    self._sequence_length = int(self.params['sequence_length'])
    self._peek_at_dataset()    

  def reshape(self, bottom, top):
    top[0].reshape(self._batch_size, 3) # image_id, image_w, image_h
    top[1].reshape(self._batch_size, self._sequence_length) # input_sentence
    top[2].reshape(self._batch_size, self._sequence_length) # target_sentence
    top[3].reshape(self._batch_size, 1) # num boxes
    if self._first_box_is_pool:
      top[4].reshape(self._batch_size, self._max_boxes+1, 4) # boxes
      top[5].reshape(self._batch_size, self._max_boxes+1, self._feature_len) # features
    else:
      top[4].reshape(self._batch_size, self._max_boxes, 4) # boxes
      top[5].reshape(self._batch_size, self._max_boxes, self._feature_len) # features
    
  def forward(self, bottom, top):
    """ Outputs image_id, input_sentence, target_sentence, boxes, features """
    if not hasattr(self, '_cur'):
      self.load_dataset()
    db_inds = self._get_next_minibatch_inds()
    top[1].data[...] = self._end_of_sequence
    top[2].data[...] = self._ignore_label
    top[4].data[...] = 0
    top[5].data[...] = 0
    for i,ix in enumerate(db_inds):
      item = self._data[ix]
      top[0].data[i] = [item['image_id'], item['image_w'], item['image_h']]
      caption = random.choice(item['captions'])  
      top[1].data[i,1:min(self._sequence_length,len(caption)+1)] = \
            caption[:self._sequence_length-1] # input_sentence
      top[2].data[i,:min(self._sequence_length,len(caption)+1)] = \
            (caption + [self._end_of_sequence])[:self._sequence_length] # target_sentence
      top[3].data[i] = item['num_boxes']
      if self._first_box_is_pool:
        top[4].data[i,1:item['num_boxes']+1,:] = item['boxes'] 
        top[5].data[i,0,:self._feature_len] = np.mean(item['features'], axis=0)
        top[5].data[i,1:item['num_boxes']+1,:self._feature_len] = item['features']
      else:
        top[4].data[i,:item['num_boxes'],:] = item['boxes']
        top[5].data[i,:item['num_boxes'],:self._feature_len] = item['features']
    top[1].data[:,0] = self._end_of_sequence

    
