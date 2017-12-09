#!/usr/bin/env python
"""
decode.py is a beam search decoder for coco captioning.
"""

import numpy as np
import pandas as pd
import os
import argparse
import time
import sys
import math
import json
import re
import string

sys.path.append('./python/')
import caffe

sys.path.append('python/caffe/proto')
import caffe_pb2
from caffe_pb2 import NetParameter, LayerParameter, DataParameter, SolverParameter, ParamSpec
from google.protobuf import text_format




def load_caption_vocab(vocab_path='./data/coco/val2014_vocab.txt'):
  print 'Loading vocab...'
  vocab = ['.']
  with open(vocab_path) as vocab_file:
    for word in vocab_file:
      vocab.append(word.strip())
  return vocab


def main():
  ''' Decode and evaluate captions '''

  # Adjustable params
  START = 150000
  STOP = 160000
  STEP = 10000

  EVALUATE=True
  GPU_ID=1
  NET_NAME='lrcn_caffenet_baseline_bow_training_schedule'
  BEAM_SIZE=5
  VERBOSE=False

  for it in range(START,STOP,STEP):

    MODEL='lrcn_iter_%d' % it
    BASE_DIR='./examples/coco_caption'
    outfile = BASE_DIR+'/results/'+NET_NAME+'_'+MODEL+'.json'

    ITERATIONS=406 # One pass through val set at minibatch=100

    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)
    
    model_def = BASE_DIR+'/test.prototxt'
    model_weights = BASE_DIR+'/snapshots/'+NET_NAME+'/'+MODEL+'.caffemodel.h5'

    # Enable BeamSearch layers to load weights
    param = NetParameter()
    with open(model_def) as f:
      text_format.Merge(f.read(), param)
    for layer in param.layer:
      if layer.HasField('beam_search_param'):
        if not layer.beam_search_param.HasField('weights'):
          layer.beam_search_param.weights = model_weights
        layer.beam_search_param.beam_size = BEAM_SIZE
    temp_model_def = model_def + '.pytmp'
    with open(temp_model_def, 'w') as f:
      f.write(str(param))

    # Load net
    net = caffe.Net(temp_model_def, caffe.TEST, weights_file=model_weights)
    caption_vocab = load_caption_vocab()
    id_to_caption = {}
    for it in range(ITERATIONS):
      if it % 100 == 0:
        print 'Iteration %d' % it
      # Outputs caption, image_id and log_prob
      output = net.forward()
      caption_blob = output['caption']
      batch_size = caption_blob.shape[0] 
      beam_size = caption_blob.shape[1]
      sequence_length = caption_blob.shape[2]
      for n in range(batch_size):
        best_caption = ''
        highest_log_p = float('-inf')
        for b in range(beam_size):
          caption_words = []
          caption = ""
          w = 0
          while w < sequence_length:
            next_word = caption_vocab[int(caption_blob[n][b][w])]
            caption_words.append(next_word)
            if w == 0:
              next_word = next_word[0].upper() + next_word[1:]
            if w > 0 and next_word not in ',.':
              caption += ' '
            if next_word[0] == '"': # Escape double quotes
              next_word = '\"' + next_word[1:]
            caption += next_word
            w = w+1
            if caption[-1] == '.':
              break
          p = output['log_prob'][n][b]
          if TAG_WEIGHT > 0.0:
            tag_log_prob = sys.float_info.min
            tags = vocab_filter.extract_tags(caption_words)
            for tag in tags:
              if tag in tag_vocab:
                tag_log_prob += tag_probs[tag_vocab[tag]]
            tag_log_prob = math.log(tag_log_prob)
            total_p = TAG_WEIGHT * tag_log_prob + (1.0 - TAG_WEIGHT) * p
            if VERBOSE:
              print 'Caption: "%s" Log Prob:%f Tag Prob:%f' %(caption, p, tag_log_prob)
          else:
            total_p = p
            if VERBOSE:
              print 'Caption: "%s" Log Prob:%f' %(caption, p)
          if total_p > highest_log_p:
            highest_log_p = total_p
            best_caption = caption
        id_to_caption[output['image_id'][n]] = (best_caption,highest_log_p)

    generated_captions = []
    for image_id, (caption,log_p) in id_to_caption.iteritems():
      generated_captions.append({
        'image_id' : int(image_id),
        'caption' : caption,
        'log_prob' : float(log_p)
      })

    with open(outfile, 'w') as out:
      json.dump(generated_captions, out, indent = 2, sort_keys=True)


if __name__ == "__main__":

  #outfile = './examples/coco_caption/results/lrcn_caffenet_baseline_letterwise_v1_lrcn_iter_20000.json'
  #s = Scorer()
  #s.score(outfile, save_results=True)

  main()


