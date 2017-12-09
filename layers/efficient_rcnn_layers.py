import caffe
import numpy as np
import ast
import base64
import csv
import random
import sys
import json
import atexit
from collections import defaultdict

csv.field_size_limit(sys.maxsize)
np.random.seed()
random.seed()

# Memory efficient version of rcnn_layers.py. If you have lots of RAM, 
# training with rcnn_layers.py is slightly faster with better minibatch randomization.


class RCNNTestDataLayer(caffe.Layer):
  """
  Data layer that outputs pre-computed faster RCNN detections.
  """
    
  def _peek_at_dataset(self):
    """ Establish the feature length. """
    with open(self.params['feature_sources'][0]) as tsvfile:
      reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames = self._fieldnames)
      for item in reader:
        item['features'] = np.frombuffer(base64.decodestring(item['features']), 
            dtype=np.float32).reshape((int(item['num_boxes']),-1))
        self._feature_len = item['features'].shape[1]
        break
          
  def _load_params(self):
    self._fieldnames = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
    self.params = ast.literal_eval(self.param_str)
    self._batch_size = int(self.params['batch_size'])
    self._first_box_is_pool = bool(self.params['pool']) # First feature is average pool
    if 'max_boxes' in self.params:
      self._max_boxes = int(self.params['max_boxes'])
    else:
      self._max_boxes = 100     
    self._src_ix = 0
    self._infile = None

  def _open_next_source(self):
    if not self._infile:
      self._infile = open(self._sources[self._src_ix], "r+b")
    elif len(self._sources) > 1:
      # Close existing file
      self._infile.close()
      # Increment source
      self._src_ix += 1
      if self._src_ix >= len(self._sources):
        self._src_ix = 0
      # Open next file
      self._infile = open(self._sources[self._src_ix], "r+b")
    else:
      self._infile.seek(0)

  def _read_next(self):
    if not self._infile:
      self._open_next_source()
    next = self._infile.readline()
    if not next:
      self._open_next_source()
      next = self._infile.readline()
    return next

  def load_dataset(self, rank=0, num_gpus=1):
    if not hasattr(self, '_sources'):
      self._load_params()
      self._sources = self.params['feature_sources'][rank::num_gpus]
      def cleanup():
        if self._infile:
          self._infile.close()
      atexit.register(cleanup)
  
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
  
  def _next_minibatch(self):
    batch = []
    for i in range(self._batch_size):
      batch.append(self._read_next())
    return batch

  def forward(self, bottom, top):
    """ Output image_id, num_boxes, boxes, features. """
    self.load_dataset()
    top[2].data[...] = 0
    top[3].data[...] = 0
    for i,row in enumerate(self._next_minibatch()):
      image_id, image_w, image_h, num_boxes, boxes, features = row.split('\t')
      image_id = int(image_id)
      image_w = int(image_w)
      image_h = int(image_h)
      num_boxes = int(num_boxes)
      boxes = np.frombuffer(base64.decodestring(boxes), dtype=np.float32).reshape((num_boxes,-1))
      features = np.frombuffer(base64.decodestring(features), dtype=np.float32).reshape((num_boxes, self._feature_len))
      top[0].data[i] = [image_id, image_w, image_h]
      top[1].data[i] = num_boxes
      if self._first_box_is_pool:
        if boxes.shape[1] == 4:
          top[2].data[i,1:num_boxes+1,:] = boxes
        top[3].data[i,0,:self._feature_len] = np.mean(features, axis=0)
        top[3].data[i,1:num_boxes+1,:self._feature_len] = features
      else:
        if boxes.shape[1] == 4:
          top[2].data[i,:num_boxes,:] = boxes
        top[3].data[i,:num_boxes,:self._feature_len] = features

  def backward(self, top, propagate_down, bottom):
    """This layer does not propagate gradients."""
    pass    


class RCNNCaptionTrainDataLayer(RCNNTestDataLayer):
  """
  Data layer that outputs pre-computed faster RCNN detections and captions
  """
    
  def load_dataset(self, rank=0, num_gpus=1):
    # Load image features
    super(RCNNCaptionTrainDataLayer, self).load_dataset(rank, num_gpus)
    if not hasattr(self, '_id_to_caption'):
      # Load captions
      self._id_to_caption = defaultdict(list)
      for src in self.params['caption_sources']:
        print 'Loading captions from: %s' % src
        with open(src) as txtfile:
          for line in txtfile.readlines():
            image_id = int(line.split('.jpg')[0].split('_')[-1])
            seq = [int(w) for w in line.split(' | ')[-1].split()]
            self._id_to_caption[image_id].append(seq)
        print 'Loaded %d image ids' % len(self._id_to_caption)
          
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
    self.load_dataset()
    top[1].data[...] = self._end_of_sequence
    top[2].data[...] = self._ignore_label
    top[4].data[...] = 0
    top[5].data[...] = 0
    for i,row in enumerate(self._next_minibatch()):
      try:
        image_id, image_w, image_h, num_boxes, boxes, features = row.split('\t')
      except ValueError as e:
        print e
        print row
      image_id = int(image_id)
      image_w = int(image_w)
      image_h = int(image_h)
      num_boxes = int(num_boxes)
      boxes = np.frombuffer(base64.decodestring(boxes), dtype=np.float32).reshape((num_boxes,-1))
      features = np.frombuffer(base64.decodestring(features), dtype=np.float32).reshape((num_boxes, self._feature_len))
      top[0].data[i] = [image_id, image_w, image_h]
      caption = random.choice(self._id_to_caption[image_id])  
      top[1].data[i,1:min(self._sequence_length,len(caption)+1)] = \
            caption[:self._sequence_length-1] # input_sentence
      top[2].data[i,:min(self._sequence_length,len(caption)+1)] = \
            (caption + [self._end_of_sequence])[:self._sequence_length] # target_sentence
      top[3].data[i] = num_boxes
      if self._first_box_is_pool:
        if boxes.shape[1] == 4:
          top[4].data[i,1:num_boxes+1,:] = boxes
        top[5].data[i,0,:self._feature_len] = np.mean(features, axis=0)
        top[5].data[i,1:num_boxes+1,:self._feature_len] = features
      else:
        if boxes.shape[1] == 4:
          top[4].data[i,:num_boxes,:] = boxes
        top[5].data[i,:num_boxes,:self._feature_len] = features
    top[1].data[:,0] = self._end_of_sequence
    
