import caffe
import numpy as np
import ast



class NMSLayer(caffe.Layer):
  """
  Layer to sample from image region candidates using Non-Max Supression (NMS). This layer
  doubles the minibatch size. The first half of each minibatch corresponds to standard
  (greedy) NMS, where the argmax next region is chosen after each suppression step. The
  second half of the minibatch corresponds to a sampled NMS process. The first region_score
  is taken as end of sequence (length must be one greater than boxes / features).
  """
  def setup(self, bottom, top):
    assert len(bottom) == 4, \
        "Inputs 4 blobs (total_num_boxes, all_boxes, all_features, region_scores)."    
    assert len(top) == 4, \
        "Outputs 4 blobs (num_boxes, boxes, features, sample_indices)."        
    self.params = ast.literal_eval(self.param_str)
    self._nms_thresh = float(self.params['nms_thresh'])
    self._max_boxes = int(self.params['max_boxes'])
    self._min_boxes = int(self.params['min_boxes'])
    self._sample = bool(self.params['sample'])
    self._pool = 1 if self.params['pool'] else 0 # First output feature is average pool
    self._end_of_sequence = 0
    if 'loss_weight' in self.params:
      self._loss_weight = float(self.params['loss_weight'])
    else:
      self._loss_weight = 1.0
    
  def reshape(self, bottom, top):
    self._batch_size = bottom[0].data.shape[0]
    self._num_boxes = bottom[3].data.shape[1]
    self._feature_len = bottom[2].data.shape[2]
    if self._sample:
      top[0].reshape(2*self._batch_size, 1) # num boxes
      top[1].reshape(2*self._batch_size, self._pool+self._max_boxes, 4) # boxes
      top[2].reshape(2*self._batch_size, self._pool+self._max_boxes, self._feature_len) # features
      top[3].reshape(2*self._batch_size, self._max_boxes) # sample_indices
    else: # Argmax output only
      top[0].reshape(self._batch_size, 1) # num boxes
      top[1].reshape(self._batch_size, self._pool+self._max_boxes, 4) # boxes
      top[2].reshape(self._batch_size, self._pool+self._max_boxes, self._feature_len) # features
      top[3].reshape(self._batch_size, self._max_boxes) # sample_indices

  def forward(self, bottom, top):
    ''' Perform Non-Max Supression '''
    x1 = bottom[1].data[:, :, 0]
    y1 = bottom[1].data[:, :, 1]
    x2 = bottom[1].data[:, :, 2]
    y2 = bottom[1].data[:, :, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    top[0].data[...] = 0
    top[1].data[...] = 0
    top[2].data[...] = 0
    top[3].data[...] =-1

    # Shift and exponentiate scores
    e = np.exp(bottom[3].data - np.max(bottom[3].data, axis=1).reshape(-1,1)) # e.g. (Nx300)
    for k,raw_num_boxes in enumerate(bottom[0].data):
      e[k,int(raw_num_boxes[0])+1:] = 0 # Exclude scores beyond softmax boundary (allow for end of sequence)
    if self._sample:
      e = np.vstack((e,e))
      self.diff = np.zeros(bottom[3].data.shape)
    step = 0
    while step < self._num_boxes and step < self._max_boxes:
      # Compute softmax of suppressed scores
      if step < self._min_boxes:
        p = e.copy()
        p[:,0] = 0
        p /= np.sum(p, axis=1).reshape(-1,1) # eos prob equal to zero
      else:
        p = e / np.sum(e, axis=1).reshape(-1,1)
      p = np.nan_to_num(p)
      # Argmax region selection
      i = p[:self._batch_size].argmax(axis=1)
      n = np.arange(len(i))
      valid_i = np.where(p[n,i]>0)[0]
      i = i[valid_i]
      n_i = n[valid_i]
      if self._sample:
        # Sampled region selection
        s = []
        n_s = []
        for k,probs in enumerate(p[self._batch_size:]):
          if sum(probs) > 0:
            s.append(np.random.choice(np.arange(len(probs)), p=probs))
            n_s.append(k)
        s = np.array(s, dtype=int)
        n_s = np.array(n_s, dtype=int)
        # Accumulate diffs for sampled regions
        self.diff[n_s] += p[n_s+self._batch_size]
        self.diff[n_s,s] -= 1.0
      # Zero all probs for ended sequences (for next iter)
      e[n_i[np.where(i == self._end_of_sequence)]] = 0
      if self._sample:
        e[self._batch_size+n_s[np.where(s == self._end_of_sequence)]] = 0
      # Remove EOS from selections, since no further output or NMS is required
      n_i = n_i[np.where(i != self._end_of_sequence)]
      i = i[i != self._end_of_sequence]
      if self._sample:
        n_s = n_s[np.where(s != self._end_of_sequence)]
        s = s[s != self._end_of_sequence]
      top[0].data[n_i] += 1 # num boxes
      top[1].data[n_i, self._pool + step, :] = bottom[1].data[n_i, i-1, :] # argmax boxes
      top[2].data[n_i, self._pool + step, :] = bottom[2].data[n_i, i-1, :] # argmax features
      top[3].data[n_i, step] = i-1 # argmax indices
      if self._sample:
        top[0].data[self._batch_size+n_s] += 1 # num boxes
        top[1].data[self._batch_size+n_s, self._pool + step, :] = bottom[1].data[n_s, s-1, :] # sample boxes
        top[2].data[self._batch_size+n_s, self._pool + step, :] = bottom[2].data[n_s, s-1, :] # sample features
        top[3].data[self._batch_size+n_s, step] = s-1 # sample indices
      #argmax NMS
      xx1 = np.maximum(x1[n_i, i-1].reshape(-1,1), x1[n_i])
      yy1 = np.maximum(y1[n_i, i-1].reshape(-1,1), y1[n_i])
      xx2 = np.minimum(x2[n_i, i-1].reshape(-1,1), x2[n_i])
      yy2 = np.minimum(y2[n_i, i-1].reshape(-1,1), y2[n_i])
      w = np.maximum(0.0, xx2 - xx1 + 1)
      h = np.maximum(0.0, yy2 - yy1 + 1)
      inter = w * h
      ovr = inter / (areas[n_i, i-1].reshape(-1,1) + areas[n_i] - inter)
      x,y = np.where(ovr >= self._nms_thresh)
      e[n_i[x],y+1] = 0
      if self._sample:
        #sample NMS
        xx1 = np.maximum(x1[n_s, s-1].reshape(-1,1), x1[n_s])
        yy1 = np.maximum(y1[n_s, s-1].reshape(-1,1), y1[n_s])
        xx2 = np.minimum(x2[n_s, s-1].reshape(-1,1), x2[n_s])
        yy2 = np.minimum(y2[n_s, s-1].reshape(-1,1), y2[n_s])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[n_s, s-1].reshape(-1,1) + areas[n_s] - inter)
        x,y = np.where(ovr >= self._nms_thresh)
        e[self._batch_size+n_s[x],y+1] = 0

      step += 1

    if self._sample:
      # Normalize gradient by number of boxes
      self.diff /= top[0].data[self._batch_size:]

    if self._pool:
      for k in range(top[2].data.shape[0]):
        top[2].data[k,0,:] = np.mean(top[2].data[k,1:int(top[0].data[k])+1,:], axis=0)

  def backward(self, top, propagate_down, bottom):
    if self._sample:
      assert hasattr(self, 'logp')
      self.weights = self.logp[self._batch_size:] - self.logp[:self._batch_size]
      bottom[3].diff[...] = self.weights.reshape(-1,1) * self.diff * self._loss_weight

  def set_logp_blob(self, data_blob):
    ''' Hack to read log probablilities of each sequence. '''
    self.logp = data_blob



class SumLogProbLayer(caffe.Layer):
  """
  Sum the log probability of each training sequence. 
  """
  def setup(self, bottom, top):
    assert len(bottom) == 2, "Inputs 2 blobs: target_sentence (NxT), logp (NxTxV)"
    assert len(top) == 1
    self.params = ast.literal_eval(self.param_str)
    self._end_of_sequence = float(self.params['end_of_sequence'])
    
  def reshape(self, bottom, top):
    self._batch_size = bottom[0].data.shape[0]
    self._sequence_len = bottom[0].data.shape[1]
    assert bottom[1].data.shape[0] == self._batch_size
    assert bottom[1].data.shape[1] == self._sequence_len
    top[0].reshape(self._batch_size)

  def forward(self, bottom, top):
    # Calculate log probablity of each sequence
    ended = np.zeros((self._batch_size,1))
    top[0].data[...] =  0
    for t in range(self._sequence_len):
      target_words = bottom[0].data[:, t].astype(int)
      logp = bottom[1].data[:, t]
      next_logp = logp[np.arange(len(target_words)), target_words]
      top[0].data[np.where(ended==0)[0]] += next_logp[np.where(ended==0)[0]]
      ended[np.where(target_words==self._end_of_sequence)[0]] = 1

  def backward(self, top, propagate_down, bottom):
    pass






