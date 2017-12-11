#!/usr/bin/python


# This scripts scaffolds prototxt files defining models and training configs
# for various experiments


import sys
import random
import subprocess
import itertools
import argparse
import numpy as np
import os
import stat
sys.path.append('./external/caffe/python/caffe/proto'); import caffe_pb2

from caffe_pb2 import NetParameter, LayerParameter, DataParameter, SolverParameter, ParamSpec, TRAIN, TEST

  
def add_weight_filler(param):
  param.type = "gaussian"
  param.std = 0.01
      
def add_bias_filler(param, value=0):
  param.type = 'constant'
  param.value = value

class CreateNet(object):
  ''' Class to scaffold out caption nets in Caffe prototxt '''

  def get_data_layers(self, param, mode='train'):
    ''' Mode can be 'train','test' or 'scst' for self-critical sequence training. '''
    net = NetParameter()
    
    if mode == 'test':
      data = net.layer.add()
      data.type = "Python"
      data.name = "test_data"
      data.top.append("image_id")
      data.top.append("num_boxes") # (b, 1) - excludes the pooled box in count
      data.top.append("boxes") # (b, max_boxes, 4)
      data.top.append("features") # (b, max_boxes, 2048)
      data.python_param.module = "rcnn_layers"
      data.python_param.layer = "RCNNTestDataLayer"
      data.python_param.param_str = str({
        'batch_size': param['test_batch_size'],
        'feature_sources' : param['test_feature_sources'],
        'pool': True,
        'max_boxes' : param['max_att_features']
      })
    
    elif mode == 'train':
      data = net.layer.add()
      data.type = "Python"
      data.name = "sequence_data"
      data.top.append("image_id")
      data.top.append("input_sentence")
      data.top.append("target_sentence")
      data.top.append("num_boxes") # (b, 1) - excludes the pooled box in count
      data.top.append("boxes") # (b, max_boxes, 4)
      data.top.append("features") # (b, max_boxes, 2048)
      data.python_param.module = "rcnn_layers"
      data.python_param.layer = "RCNNCaptionTrainDataLayer"
      data.python_param.param_str = str({
        'end_of_sequence': param['end_of_sequence'],
        'ignore_label': param['ignore_label'],
        'sequence_length': param['max_length'],
        'batch_size': param['train_batch_size'],
        'caption_sources' : param['train_caption_sources'],
        'feature_sources' : param['train_feature_sources'],
        'pool': True,
        'max_boxes' : param['max_att_features']
      })
    
    elif mode == 'scst':
      data = net.layer.add()
      data.type = "Python"
      data.name = "train_data"
      data.top.append("image_id")
      data.top.append("num_boxes") # (b, 1) - excludes the pooled box in count
      data.top.append("boxes") # (b, max_boxes, 4)
      data.top.append("features") # (b, max_boxes, 2048)
      data.python_param.module = "rcnn_layers"
      data.python_param.layer = "RCNNTestDataLayer"
      data.python_param.param_str = str({
        'batch_size': param['train_batch_size'],
        'feature_sources' : param['train_feature_sources'],
        'pool': True,
        'max_boxes' : param['max_att_features']
      })
    
    else:
      raise ValueError('unknown data layer mode')
    
    slice_layer = net.layer.add()
    slice_layer.name = "feature_slice_layer"
    slice_layer.type = "Slice"
    slice_layer.slice_param.slice_dim = 1
    slice_layer.slice_param.slice_point.append(1)
    slice_layer.bottom.append("features")
    slice_layer.top.append("avg_pool") # batch_size x 1 x cnn_filters
    slice_layer.top.append("spatial_features") # batch_size x att_features x cnn_filters
    
    flatten_layer = net.layer.add()
    flatten_layer.name = "context"
    flatten_layer.type = "Flatten"
    flatten_layer.bottom.append("avg_pool") # batch_size 2048
    flatten_layer.top.append(flatten_layer.name) # batch_size x 2048
    flatten_layer.flatten_param.axis = 1

    fc_layer = net.layer.add()
    fc_layer.name = "fc"
    fc_layer.type = "InnerProduct"
    fc_layer.bottom.append("spatial_features") # batch_size x att_features x cnn_filters
    fc_layer.top.append(fc_layer.name) # batch_size x att_features x att_hidden_units 
    fc_layer.inner_product_param.num_output = param['att_hidden_units']
    fc_layer.inner_product_param.axis = 2
    fc_layer.inner_product_param.bias_term = False
    add_weight_filler(fc_layer.inner_product_param.weight_filler)

    if mode == 'test' or mode == 'scst':
      # Tile inputs for the number of beams and reshape into batchsize
      tile_layer = net.layer.add()
      tile_layer.name = "tile_context"
      tile_layer.top.append(tile_layer.name)
      tile_layer.bottom.append("context")
      tile_layer.type = "Tile"
      tile_layer.tile_param.tiles = param['test_beam_size']
      tile_layer.tile_param.axis = 1

      reshape_layer = net.layer.add()
      reshape_layer.name = "beam_context"
      reshape_layer.type = "Reshape"
      reshape_layer.bottom.append("tile_context")
      reshape_layer.top.append(reshape_layer.name)
      reshape_layer.reshape_param.shape.dim.append(-1)
      reshape_layer.reshape_param.shape.dim.append(param['cnn_filters'])
      
      tile_layer = net.layer.add()
      tile_layer.name = "tile_spatial_features"
      tile_layer.top.append(tile_layer.name)
      tile_layer.bottom.append("spatial_features")
      tile_layer.type = "Tile"
      tile_layer.tile_param.tiles = param['test_beam_size']
      tile_layer.tile_param.axis = 1

      reshape_layer = net.layer.add()
      reshape_layer.name = "beam_spatial_features"
      reshape_layer.type = "Reshape"
      reshape_layer.bottom.append(tile_layer.name)
      reshape_layer.top.append(reshape_layer.name)
      reshape_layer.reshape_param.shape.dim.append(-1)
      reshape_layer.reshape_param.shape.dim.append(param['max_att_features'])
      reshape_layer.reshape_param.shape.dim.append(param['cnn_filters'])
      
      tile_layer = net.layer.add()
      tile_layer.name = "tile_fc"
      tile_layer.top.append(tile_layer.name)
      tile_layer.bottom.append("fc")
      tile_layer.type = "Tile"
      tile_layer.tile_param.tiles = param['test_beam_size']
      tile_layer.tile_param.axis = 1

      reshape_layer = net.layer.add()
      reshape_layer.name = "beam_fc"
      reshape_layer.type = "Reshape"
      reshape_layer.bottom.append(tile_layer.name)
      reshape_layer.top.append(reshape_layer.name)
      reshape_layer.reshape_param.shape.dim.append(-1)
      reshape_layer.reshape_param.shape.dim.append(param['max_att_features'])
      reshape_layer.reshape_param.shape.dim.append(param['att_hidden_units'])
      
      tile_layer = net.layer.add()
      tile_layer.name = "tile_num_boxes"
      tile_layer.top.append(tile_layer.name)
      tile_layer.bottom.append("num_boxes")
      tile_layer.type = "Tile"
      tile_layer.tile_param.tiles = param['test_beam_size']
      tile_layer.tile_param.axis = 1

      reshape_layer = net.layer.add()
      reshape_layer.name = "beam_num_boxes"
      reshape_layer.type = "Reshape"
      reshape_layer.bottom.append(tile_layer.name)
      reshape_layer.top.append(reshape_layer.name)
      reshape_layer.reshape_param.shape.dim.append(-1)
      reshape_layer.reshape_param.shape.dim.append(1) 
    return net


  def get_scst_net(self, param):
    net = NetParameter()

    beam_layer = net.layer.add()
    beam_layer.name = "beam"
    beam_layer.type = "BeamSearch"
    beam_layer.bottom.append("num_boxes")
    beam_layer.bottom.append("spatial_features")
    beam_layer.bottom.append("fc")
    beam_layer.bottom.append("context")
    beam_layer.top.append("caption")
    beam_layer.top.append("log_prob")
    beam_layer.top.append("log_prob_sequence")
    bs = beam_layer.beam_search_param
    bs.beam_size = param['test_beam_size']
    bs.sequence_length = param['max_length']
    bs.end_of_sequence = param['end_of_sequence']
    for word in param['allowed_multiple']:
      bs.allowed_multiple.append(word)
    for i in range(param['num_lstm_stacks']):
      # Previous hidden state
      rc = bs.recurrent_connection.add()
      rc.src = 'lstm%d_hidden0' % i
      rc.dest = 'lstm%d_hidden_prev' % i
      # Previous mem cell
      rc = bs.recurrent_connection.add()
      rc.src = 'lstm%d_mem_cell0' % i
      rc.dest = 'lstm%d_mem_cell_prev' % i
    bs.beam_search_connection.src = 'logp_0'
    bs.beam_search_connection.dest = 'input'
    for pname in ["embed_param", "lstm0_param_0", "lstm0_param_1", "hidden_att_param_0",
          "predict_att_param_0", "lstm1_param_0", "lstm1_param_1", "predict_param_0", 
          "predict_param_1"]:
      p = beam_layer.param.add()
      p.name = pname # Share weights

    inner_net = bs.net_param
    input_layer = inner_net.layer.add()
    input_layer.name = "input"
    input_layer.type = "Input"
    input_layer.top.append("num_boxes")
    input_layer.top.append("spatial_features")
    input_layer.top.append("fc")
    input_layer.top.append("context")
    input_layer.top.append(input_layer.name)
    blob_shape = input_layer.input_param.shape.add()
    blob_shape.dim.append(param['train_batch_size'])
    blob_shape.dim.append(1)   
    blob_shape = input_layer.input_param.shape.add()
    blob_shape.dim.append(param['train_batch_size'])
    blob_shape.dim.append(param['max_att_features'])
    blob_shape.dim.append(param['cnn_filters'])
    blob_shape = input_layer.input_param.shape.add()
    blob_shape.dim.append(param['train_batch_size'])
    blob_shape.dim.append(param['max_att_features'])
    blob_shape.dim.append(param['att_hidden_units'])
    blob_shape = input_layer.input_param.shape.add()
    blob_shape.dim.append(param['train_batch_size'])
    blob_shape.dim.append(param['cnn_filters'])
    blob_shape = input_layer.input_param.shape.add()
    blob_shape.dim.append(param['train_batch_size'])
    blob_shape.dim.append(1)
    
    max_length = param['max_length']
    param['max_length'] = 1
    inner_net = self.get_net(param, param['test_batch_size'], 
        net=inner_net, mode="scst_decode")
    
    silence_layer = net.layer.add()
    silence_layer.name = "silence_bs"
    silence_layer.type = "Silence"
    silence_layer.bottom.append("log_prob")
    silence_layer.bottom.append("log_prob_sequence")
    
    scst_layer = net.layer.add()
    scst_layer.type = "Python"
    scst_layer.name = "scst"
    scst_layer.bottom.append("image_id")
    scst_layer.bottom.append("caption")
    scst_layer.propagate_down.append(False)
    scst_layer.propagate_down.append(False)
    scst_layer.top.append("score_weights")
    scst_layer.top.append("input_sentence")
    scst_layer.top.append("target_sentence")
    scst_layer.top.append("mean_score")
    scst_layer.python_param.module = "scst_layers"
    scst_layer.python_param.layer = "SCSTLayer"
    scst_layer.python_param.param_str = str({
      'vocab_path' : param['data_dir'] + param['vocab_file'],
      'gt_caption_paths' : param['gt_caption_paths'],
      'end_of_sequence' : param['end_of_sequence'],
      'ignore_label' : param['ignore_label']
    })
    
    # Add rest of training network
    param['max_length'] = max_length
    net = self.get_net(param, param['test_batch_size']*param['test_beam_size'], 
        net=net, mode="scst_train")
        
    return net
    
    
  def get_net(self, param, batch_size, loss_weight=None, net=None, mode="train"):
    ''' Mode can be 'train','test','scst_decode' or 'scst_train'. '''
  
    if not net:
      net = NetParameter()

    if mode=="test":
      dummy_layer = net.layer.add()
      dummy_layer.name = "input"
      dummy_layer.top.append(dummy_layer.name)
      dummy_layer.type = "DummyData"
      filler = dummy_layer.dummy_data_param.data_filler.add()
      # Note that DataLayer uses the end_of_sequence to start sequences,
      # so we will do the same
      filler.value = param['end_of_sequence']
      blob_shape = dummy_layer.dummy_data_param.shape.add()
      blob_shape.dim.append(batch_size)
      blob_shape.dim.append(1)
    if mode == "train" or mode == "scst_train":
      input_slice_layer = net.layer.add()
      input_slice_layer.name = "input_slice"
      input_slice_layer.type = "Slice"
      input_slice_layer.slice_param.slice_dim = 1
      input_slice_layer.bottom.append('input_sentence')
      for i in range(param['max_length']):
        input_slice_layer.top.append("input" if i == 0 else "input_%d" % i)
        if i != 0:
          input_slice_layer.slice_param.slice_point.append(i)

    for i in range(param['max_length']):
      if i == 0:

        for j in range(param['num_lstm_stacks']):
          dummy_layer = net.layer.add()
          dummy_layer.name = 'lstm%d_hidden_prev' % j
          dummy_layer.top.append(dummy_layer.name)
          dummy_layer.type = "DummyData"
          blob_shape = dummy_layer.dummy_data_param.shape.add()
          blob_shape.dim.append(batch_size)
          blob_shape.dim.append(param['lstm_num_cells'])

          dummy_mem_cell = net.layer.add()
          dummy_mem_cell.name = 'lstm%d_mem_cell_prev' %j
          dummy_mem_cell.top.append(dummy_mem_cell.name)
          dummy_mem_cell.type = "DummyData"
          blob_shape = dummy_mem_cell.dummy_data_param.shape.add()
          blob_shape.dim.append(batch_size)
          blob_shape.dim.append(param['lstm_num_cells'])

      for j in range(param['num_lstm_stacks']):

        if j == 0:
          embed_layer = net.layer.add()
          embed_layer.name = "embedding" if i == 0 else "embedding_%d" % i
          embed_layer.type = "Embed"
          embed_layer.bottom.append("input" if i == 0 else "input_%d" % i)
          embed_layer.propagate_down.append(False)
          embed_layer.top.append(embed_layer.name)
          embed_layer.embed_param.bias_term = False
          embed_layer.embed_param.input_dim = param['vocab_size']
          embed_layer.embed_param.num_output = param['lstm_num_cells']
          add_weight_filler(embed_layer.embed_param.weight_filler)
          p = embed_layer.param.add()
          p.name = 'embed_param' # Share weights
          
        if j != 0:
          # Set up attention mechanism

          inner_product_layer = net.layer.add()
          inner_product_layer.name = "hidden_att_%d" % i
          inner_product_layer.bottom.append(lstm_output_blob) # batch_size x lstm_num_cells
          inner_product_layer.top.append(inner_product_layer.name) # batch_size x att_hidden_units
          inner_product_layer.type = "InnerProduct"
          inner_product_layer.inner_product_param.num_output = param['att_hidden_units']
          inner_product_layer.inner_product_param.bias_term = False
          add_weight_filler(inner_product_layer.inner_product_param.weight_filler)
          p = inner_product_layer.param.add()
          p.name = 'hidden_att_param_0' # Share weights

          tile_layer = net.layer.add()
          tile_layer.name = "tile_hidden_att_%d" % i
          tile_layer.top.append(tile_layer.name) # batch_size x (att_features x att_hidden_units)
          tile_layer.bottom.append(inner_product_layer.name)
          tile_layer.type = "Tile"
          tile_layer.tile_param.axis = 1
          tile_layer.tile_param.tiles = param['max_att_features']

          reshape_layer = net.layer.add()
          reshape_layer.name = "tile_hidden_reshape_%d" % i
          reshape_layer.type = "Reshape"
          reshape_layer.bottom.append(tile_layer.name)
          reshape_layer.top.append(reshape_layer.name)
          reshape_layer.reshape_param.shape.dim.append(0) # Batch size
          reshape_layer.reshape_param.shape.dim.append(-1)
          reshape_layer.reshape_param.shape.dim.append(param['att_hidden_units'])

          sum_layer = net.layer.add()
          sum_layer.name = "sum_hidden_att_%d" % i
          sum_layer.top.append(sum_layer.name) # batch_size x att_features x att_hidden_units
          if mode == 'test' or mode == 'scst_train':
            sum_layer.bottom.append("beam_fc") # batch_size x att_features x att_hidden_units
          else:
            sum_layer.bottom.append("fc") # batch_size x att_features x att_hidden_units
          sum_layer.bottom.append(reshape_layer.name) # batch_size x att_features x att_hidden_units
          sum_layer.type = "Eltwise"
          sum_layer.eltwise_param.operation = sum_layer.eltwise_param.SUM

          tanh_layer = net.layer.add()
          tanh_layer.name = "hidden_tanh_%d" % i
          tanh_layer.top.append(sum_layer.name)
          tanh_layer.bottom.append(sum_layer.name)
          tanh_layer.type = "TanH"

          proj_layer = net.layer.add()
          proj_layer.name = "predict_att_%d" % i
          proj_layer.type = "InnerProduct"
          proj_layer.bottom.append(sum_layer.name) # batch_size x att_features x att_hidden_units
          proj_layer.top.append(proj_layer.name) # batch_size x att_features x 1
          proj_layer.inner_product_param.num_output = 1
          proj_layer.inner_product_param.axis = 2
          proj_layer.inner_product_param.bias_term = False
          add_weight_filler(proj_layer.inner_product_param.weight_filler)          
          p = proj_layer.param.add()
          p.name = 'predict_att_param_0' # Share weights
          
          reshape_layer = net.layer.add()
          reshape_layer.name = "reshape_predict_att_%d" % i
          reshape_layer.type = "Reshape"
          reshape_layer.bottom.append(proj_layer.name)
          reshape_layer.top.append(reshape_layer.name) # batch_size x att_features
          reshape_layer.reshape_param.shape.dim.append(0) # Batch size
          reshape_layer.reshape_param.shape.dim.append(-1)

          softmax_layer = net.layer.add()
          softmax_layer.name = "att_weight_%d" % i
          softmax_layer.type = "Softmax"
          softmax_layer.bottom.append(reshape_layer.name) # batch_size x att_features
          if mode == 'test' or mode == 'scst_train':
            softmax_layer.bottom.append("beam_num_boxes") # batch_size x 1
          else:
            softmax_layer.bottom.append("num_boxes") # batch_size x 1
          softmax_layer.top.append(softmax_layer.name) # batch_size x att_features
          softmax_layer.softmax_param.axis = 1
          softmax_layer.softmax_param.engine = softmax_layer.softmax_param.CAFFE # Not implemented in CUDNN
          
          scale_layer = net.layer.add()
          scale_layer.name = "att_product_%d" % i
          scale_layer.top.append(scale_layer.name) # batch_size x att_features x input_feature_size
          if mode == 'test' or mode == 'scst_train':
            scale_layer.bottom.append("beam_spatial_features") # batch_size x att_features x input_feature_size
          else:
            scale_layer.bottom.append("spatial_features") # batch_size x att_features x input_feature_size
          scale_layer.bottom.append(softmax_layer.name) # batch_size x att_features 
          scale_layer.type = "Scale"
          scale_layer.scale_param.axis = 0
          
          permute_layer = net.layer.add()
          permute_layer.name = "permute_att_%d" % i
          permute_layer.bottom.append(scale_layer.name) # batch_size x att_features x input_feature_size
          permute_layer.top.append(permute_layer.name) # batch_size x input_feature_size x att_features
          permute_layer.type = "Permute"
          permute_layer.permute_param.order.append(0)
          permute_layer.permute_param.order.append(2)
          permute_layer.permute_param.order.append(1)          

          reduction_layer = net.layer.add()
          reduction_layer.name = "fc8_%d" % i
          reduction_layer.type = "Reduction"
          reduction_layer.bottom.append(permute_layer.name) # batch_size x input_feature_size x att_features
          reduction_layer.top.append(reduction_layer.name) # batch_size x input_feature_size
          reduction_layer.reduction_param.axis = 2

        concat_layer = net.layer.add()
        concat_layer.name = 'concat%d_t%d' % (j, i)
        concat_layer.top.append(concat_layer.name)
        concat_layer.type = "Concat"
        # Data input is either the last word or the layer below output
        if j == 0:
          concat_layer.bottom.append(embed_layer.name)
          if mode == 'test' or mode == 'scst_train':
            concat_layer.bottom.append("beam_context") # Add CNN context
          else:
            concat_layer.bottom.append("context") # Add CNN context
          # Add copy down from lstm above
          if i == 0:
            concat_layer.bottom.append('lstm%d_hidden_prev' % (j+1))
          else:
            if mode == 'test' or mode == 'scst_decode':
              concat_layer.bottom.append('lstm%d_hidden_prev%d' % (j+1, i))
            else:
              concat_layer.bottom.append('lstm%d_hidden%d' % (j+1, i-1))
        else:
          concat_layer.bottom.append('lstm%d_hidden%d' % (j - 1, i))
          concat_layer.bottom.append(reduction_layer.name) # Include attended cnn feature as next LSTM input
        # Plus either dummy or the diagonal connection from before
        if i == 0:
          concat_layer.bottom.append('lstm%d_hidden_prev' % (j))
        else:
          if mode == 'test' or mode == 'scst_decode':
            concat_layer.bottom.append('lstm%d_hidden_prev%d' % (j, i))
          else:
            concat_layer.bottom.append('lstm%d_hidden%d' % (j, i - 1))

        lstm_layer = net.layer.add()
        lstm_layer.name = 'lstm%d' % (j+1) if i==0 else 'lstm%d_t%d' % (j+1, i)
        lstm_layer.type = "LSTMNode"
        lstm_layer.lstm_param.num_cells = param['lstm_num_cells']

        add_weight_filler(lstm_layer.lstm_param.input_weight_filler)
        add_weight_filler(lstm_layer.lstm_param.input_gate_weight_filler)
        add_weight_filler(lstm_layer.lstm_param.forget_gate_weight_filler)
        add_weight_filler(lstm_layer.lstm_param.output_gate_weight_filler)
        
        add_bias_filler(lstm_layer.lstm_param.input_bias_filler)
        add_bias_filler(lstm_layer.lstm_param.input_gate_bias_filler)
        add_bias_filler(lstm_layer.lstm_param.forget_gate_bias_filler, 1)
        add_bias_filler(lstm_layer.lstm_param.output_gate_bias_filler)

        for k in range(2):
          param_spec = lstm_layer.param.add()
          param_spec.name = 'lstm%d_param_%d' % (j, k)
        lstm_output_blob = 'lstm%d_hidden%d' % (j, i)
        lstm_layer.top.append(lstm_output_blob)
        lstm_layer.top.append('lstm%d_mem_cell%d' % (j, i))
        lstm_layer.bottom.append('concat%d_t%d' % (j, i))
        lstm_layer.propagate_down.append(True)
        if i == 0:
          lstm_layer.bottom.append('lstm%d_mem_cell_prev' % j)
          lstm_layer.propagate_down.append(False)
        else:
          if mode == 'test' or mode == 'scst_decode':
            lstm_layer.bottom.append('lstm%d_mem_cell_prev%d' % (j, i))
          else:
            lstm_layer.bottom.append('lstm%d_mem_cell%d' % (j, i - 1))
          lstm_layer.propagate_down.append(True)

        if j == param['num_lstm_stacks']-1:
        
          inner_product_layer = net.layer.add()
          inner_product_layer.name = "predict" if i == 0 else "predict_%d" % i
          inner_product_layer.top.append(inner_product_layer.name)
          inner_product_layer.bottom.append(lstm_output_blob)
          inner_product_layer.type = "InnerProduct"
          inner_product_layer.inner_product_param.num_output = param['vocab_size']
          inner_product_layer.inner_product_param.axis = 1
          p = inner_product_layer.param.add()
          p.lr_mult = 1
          p.decay_mult = 1
          p.name = 'predict_param_0'
          p = inner_product_layer.param.add()
          p.lr_mult = 2
          p.decay_mult = 0
          p.name = 'predict_param_1'
          add_weight_filler(inner_product_layer.inner_product_param.weight_filler)
          add_bias_filler(inner_product_layer.inner_product_param.bias_filler)

          if mode == 'test' or mode == 'scst_decode':
            softmax_layer = net.layer.add()
            softmax_layer.name = "probs_%d" % i
            softmax_layer.type = "Softmax"
            softmax_layer.bottom.append(inner_product_layer.name)
            softmax_layer.top.append(softmax_layer.name)
            softmax_layer.softmax_param.axis = 1
            
            log_layer = net.layer.add()
            log_layer.name = "logp_%d" % i
            log_layer.type = "Log"
            log_layer.bottom.append(softmax_layer.name)
            log_layer.top.append(log_layer.name)
            
          if mode == 'test':
            # Beam search test
            bs_layer = net.layer.add()         
            bs_layer.name = "beam_search_%d" % i
            bs_layer.type = "BeamSearchNode" 
            if i > 0:
              bs_layer.bottom.append("bs_scores_%d" % (i-1))
              bs_layer.bottom.append("bs_sentence_%d" % (i-1))
            bs_layer.bottom.append(log_layer.name)
            for k in range(param['num_lstm_stacks']):
              bs_layer.bottom.append("lstm%d_hidden%d" % (k,i))
              bs_layer.bottom.append("lstm%d_mem_cell%d" % (k,i))
            
            bs_layer.top.append("log_prob" if i+1==param['max_length'] else "bs_scores_%d" % i)
            bs_layer.top.append("caption" if i+1==param['max_length'] else "bs_sentence_%d" % i)
            bs_layer.top.append("input_%d" % (i+1))
            for k in range(param['num_lstm_stacks']):
              bs_layer.top.append("lstm%d_hidden_prev%d" % (k,i+1))
              bs_layer.top.append("lstm%d_mem_cell_prev%d" % (k,i+1))
            
            bs = bs_layer.beam_search_param
            bs.beam_size = param['test_beam_size']
            bs.end_of_sequence = param['end_of_sequence']
            bs.ignore_label = param['ignore_label']
            for word in param['allowed_multiple']:
              bs.allowed_multiple.append(word)

    if mode == 'train' or mode == "scst_train":
      hidden_concat_layer = net.layer.add()
      hidden_concat_layer.type = "Concat"
      hidden_concat_layer.name = 'predict_concat'
      hidden_concat_layer.top.append(hidden_concat_layer.name)
      hidden_concat_layer.concat_param.concat_dim = 1
      for i in range(param['max_length']):
        hidden_concat_layer.bottom.append("predict" if i == 0 else "predict_%d" % i)

      reshape_layer = net.layer.add()
      reshape_layer.name = 'predict_reshape'
      reshape_layer.type = "Reshape"
      reshape_layer.bottom.append('predict_concat')
      reshape_layer.top.append(reshape_layer.name)
      reshape_layer.reshape_param.shape.dim.append(0) # Batch size
      reshape_layer.reshape_param.shape.dim.append(param['max_length'])
      reshape_layer.reshape_param.shape.dim.append(param['vocab_size'])

      word_loss_layer = net.layer.add()
      word_loss_layer.name = "cross_entropy_loss"
      word_loss_layer.type = "SoftmaxWithLoss"
      word_loss_layer.bottom.append("predict_reshape")
      word_loss_layer.bottom.append("target_sentence")
      word_loss_layer.propagate_down.append(True)
      word_loss_layer.propagate_down.append(False)
      if mode == "scst_train":
        word_loss_layer.bottom.append("score_weights")
        word_loss_layer.propagate_down.append(False)
      word_loss_layer.top.append(word_loss_layer.name)
      if loss_weight is None:
        word_loss_layer.loss_weight.append(param['max_length'])
      else:
        word_loss_layer.loss_weight.append(loss_weight)
      word_loss_layer.loss_param.ignore_label = param['ignore_label']
      word_loss_layer.softmax_param.axis = 2

      accuracy_layer = net.layer.add()
      accuracy_layer.name = "accuracy"
      accuracy_layer.type = "Accuracy"
      accuracy_layer.bottom.append("predict_reshape")
      accuracy_layer.bottom.append("target_sentence")
      accuracy_layer.top.append(accuracy_layer.name)
      accuracy_layer.accuracy_param.ignore_label = param['ignore_label']
      accuracy_layer.accuracy_param.axis = 2

    if mode != 'scst_decode':
      silence_layer = net.layer.add()
      silence_layer.name = "silence"
      silence_layer.type = "Silence"
      for j in range(param['num_lstm_stacks']):
        silence_layer.bottom.append("lstm%d_mem_cell%d" % (j, param['max_length'] - 1))
      silence_layer.bottom.append("image_id")
      silence_layer.bottom.append("boxes")
      if mode == 'test':
        silence_layer.bottom.append("input_%d" % param['max_length'])
        for j in range(param['num_lstm_stacks']):
          silence_layer.bottom.append("lstm%d_hidden_prev%d" % (j,param['max_length']))
          silence_layer.bottom.append("lstm%d_mem_cell_prev%d" % (j,param['max_length']))

    return net


  def write_solver(self, param, file_name):
    with open(file_name, 'w') as f:
      f.write(str(param))

  def write_net(self, param, file_name):
    with open(file_name, 'w') as f:
      f.write('name: "%s"\n' % param['net_name'])
      f.write(str(self.get_data_layers(param, mode='train')))   
      f.write(str(self.get_net(param, mode='train',
        batch_size = param['train_batch_size'])))
        
  def write_decoder(self, param, file_name):
    with open(file_name, 'w') as f:
      f.write('name: "%s"\n' % param['net_name'])
      f.write(str(self.get_data_layers(param, mode='test')))
      batch_size = param['test_beam_size']*param['test_batch_size']
      f.write(str(self.get_net(param, batch_size, mode='test')))

  def write_scst_net(self, param, file_name):
    with open(file_name, 'w') as f:
      f.write('name: "%s"\n' % param['net_name'])
      param['train_batch_size'] = param['test_batch_size']
      f.write(str(self.get_data_layers(param, mode='scst')))
      f.write(str(self.get_scst_net(param)))
      
  def write_train_script(self, param):
    file_name = '%s%s/train.sh' % (param['base_dir'],param['net_name'])
    with open(file_name, 'w') as f:
      f.write('''#!/bin/bash

GPU_ID=%s
BASE_DIR=%s
DATA_DIR=%s
LOG_DIR=%s
SNAPSHOT_DIR=%s
NET_NAME=%s
OUT_DIR=%s
VOCAB_FILE=%s
MAX_IT=%s
SCST_MAX_IT=%s

mkdir -p ${LOG_DIR}${NET_NAME}
mkdir -p ${SNAPSHOT_DIR}${NET_NAME}
mkdir -p ${OUT_DIR}${NET_NAME}
   
python -u external/caffe/python/train.py \
    --solver ${BASE_DIR}${NET_NAME}/solver.prototxt \
    --gpus ${GPU_ID//,/ } \
    > ${LOG_DIR}${NET_NAME}/solver.log 2<&1 
    
# Decode the cross entropy trained model
python ./scripts/beam_decode.py   --gpu ${GPU_ID:0:1}   \
    --model ${BASE_DIR}${NET_NAME}/decoder.prototxt \
    --weights=${SNAPSHOT_DIR}${NET_NAME}/lstm_iter_${MAX_IT}.caffemodel.h5 \
    --vocab ${DATA_DIR}${VOCAB_FILE} \
    --outfile ${OUT_DIR}/${NET_NAME}/iter_${MAX_IT}.json 

# Self-critical sequence training
python -u external/caffe/python/train.py \
    --solver ${BASE_DIR}${NET_NAME}/scst_solver.prototxt \
    --gpus ${GPU_ID//,/ } \
    --weights=${SNAPSHOT_DIR}${NET_NAME}/lstm_iter_${MAX_IT}.caffemodel.h5 \
    > ${LOG_DIR}${NET_NAME}/scst.log 2<&1
    
# Decode the finished model
python ./scripts/beam_decode.py   --gpu ${GPU_ID:0:1}   \
    --model ${BASE_DIR}${NET_NAME}/decoder.prototxt \
    --weights=${SNAPSHOT_DIR}${NET_NAME}/lstm_scst_iter_${SCST_MAX_IT}.caffemodel.h5 \
    --vocab ${DATA_DIR}${VOCAB_FILE} \
    --outfile ${OUT_DIR}/${NET_NAME}/scst_iter_${SCST_MAX_IT}.json


''' % (param['gpu_ids'], param['base_dir'], param['data_dir'], param['log_dir'], 
        param['snapshot_dir'], param['net_name'], param['out_dir'], 
        param['vocab_file'], param['max_iter'], param['scst_max_iter']))
    st = os.stat(file_name) # Make executable
    os.chmod(file_name, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


  def get_solver(self, param):
    solver = SolverParameter()
    solver.net = param['base_dir'] + param['net_name'] + "/net.prototxt"

    solver.base_lr = 0.01
    solver.weight_decay = 0.0005
    solver.lr_policy = "poly"
    solver.power = 1
    solver.momentum = 0.9
    solver.type = "SGD"
    solver.clip_gradients = 10

    solver.display = 100
    solver.max_iter = param['max_iter']
    solver.average_loss = 100
    solver.snapshot = param['solver_snapshot_interval']
    solver.snapshot_prefix = param['snapshot_dir']+param['net_name']+"/lstm"
    solver.snapshot_format = solver.HDF5
    solver.random_seed = param['random_seed']
    solver.iter_size = 1
    solver.layer_wise_reduce = False
    return solver
    
  
  def get_scst_solver(self, param):
    solver = self.get_solver(param)
    solver.base_lr = 0.001
    solver.lr_policy = "exp"
    solver.iter_size = 4
    solver.gamma = 0.99975
    solver.snapshot = 500
    solver.max_iter = param['scst_max_iter']
    solver.net = param['base_dir'] + param['net_name'] + "/scst_net.prototxt"
    solver.snapshot_prefix = param['snapshot_dir']+param['net_name']+"/lstm_scst"
    return solver


  def get_base_param(self, test_submission=False):
    param = {}
    param['net_name'] = 'caption_lstm'
    param['random_seed'] = 1701
    param['train_caption_sources'] = ['data/coco_splits/train_captions.txt']
    param['train_feature_sources'] = ['data/tsv/trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.%d' % i for i in range(2)]
    if test_submission:
      # Test on coco test2014
      param['test_feature_sources'] = ['data/tsv/test2014/test2014_resnet101_faster_rcnn_genome.tsv']
    else:
      # Test on karpathy test
      param['test_feature_sources'] = ['data/tsv/trainval/karpathy_test_resnet101_faster_rcnn_genome.tsv']
    param['max_iter'] = 60000
    param['scst_max_iter'] = 1000
    param['max_length'] = 20
    param['gt_caption_paths'] = ['data/coco/captions_train2014.json', 
                                 'data/coco/captions_val2014.json'] # For SCST training
    param['vocab_file']= 'trainval_vocab.txt' if test_submission else 'train_vocab.txt'
    param['vocab_size'] = 10387 if test_submission else 10010
    # Only typical stopwords are allowed to be repeated in captions:
    # a the of , on in with and an " to some at are it that ' by
    param['allowed_multiple'] = [2,5,4,15,3,6,8,7,9,13,277,11,30,16,19,27,25,119,48]
    param['end_of_sequence'] = 0 # This is the period token in the vocab file
    param['ignore_label'] = 1 # This is the unknown word token in vocab file
    param['train_batch_size'] = 50 # Remember, this is per gpu
    param['test_batch_size'] = 12
    param['test_beam_size'] = 5
    
    param['cnn_filters'] = 2048 
    param['num_lstm_stacks'] = 2
    param['lstm_num_cells'] = 1000
    param['max_att_features'] = 100
    param['att_hidden_units'] = 512
    
    param['gpu_ids'] = '0,1'
    param['base_dir'] = 'experiments/'
    param['data_dir'] = 'data/coco_splits/'
    param['snapshot_dir'] = 'snapshots/'
    param['log_dir'] = 'logs/'
    param['out_dir'] = 'outputs/'
    param['solver_snapshot_interval'] = 5000
      
    return param


def make_dir(path):
  if not os.path.exists(path):
    os.makedirs(path)


def main(builder):

  param = builder.get_base_param()
  make_dir(param['base_dir']+param['net_name'])
  print 'Creating net: %s' % param['net_name']
  builder.write_train_script(param)
  
  solver_param = builder.get_solver(param)
  file_name = '%s%s/solver.prototxt' % (param['base_dir'], param['net_name'])
  builder.write_solver(solver_param, file_name)
  
  solver_param = builder.get_scst_solver(param)
  file_name = '%s%s/scst_solver.prototxt' % (param['base_dir'], param['net_name'])
  builder.write_solver(solver_param, file_name)
  
  file_name = '%s%s/net.prototxt' % (param['base_dir'],param['net_name'])
  builder.write_net(param, file_name)
  
  file_name = '%s%s/decoder.prototxt' % (param['base_dir'],param['net_name'])
  builder.write_decoder(param, file_name)
  
  file_name = '%s%s/scst_net.prototxt' % (param['base_dir'],param['net_name'])
  builder.write_scst_net(param, file_name)

if __name__ == '__main__':
  builder = CreateNet()
  main(builder)


