import caffe
import numpy as np
import ast
import math

from cider_scorer import CiderScorer
    
class SCSTLayer(caffe.Layer):
  """
  Self-Critical Sequence Training (SCST) layer. Takes beam search and
  outputs weights for training.
  """

  def setup(self, bottom, top):
    if len(bottom) != 2:
      raise Exception("Inputs 2 bottom blobs - image_ids and captions.")
    if len(top) != 4:
      raise Exception("Outputs 3 top blobs - score_weights, input_sentence, target_sentence, mean_score.")
    params = ast.literal_eval(self.param_str)
    self._end_of_sequence = params['end_of_sequence']
    self._ignore_label = params['ignore_label']
    # Load vocab
    self._vocab = []
    with open(params['vocab_path']) as vocab_file:
      for word in vocab_file:
        self._vocab.append(word.lower().strip())
    self._cider = CiderScorer(params['gt_caption_paths'])
    
  def _translate(self, blob):
    # Results will be lower case, tokenized, without full stop
    # (to match reference tokenization)
    caption = [];
    for ix in blob:
      next_word = self._vocab[int(ix)]
      if next_word == '.':
        break
      caption.append(next_word)
    return caption
      
  def reshape(self, bottom, top):
    self._batch_size = bottom[1].shape[0]
    self._beam_size = bottom[1].shape[2]
    self._sequence_length = bottom[1].shape[3]
    top[0].reshape(self._batch_size*self._beam_size, self._sequence_length)
    top[1].reshape(self._batch_size*self._beam_size, self._sequence_length)
    top[2].reshape(self._batch_size*self._beam_size, self._sequence_length)
    top[3].reshape(1)

  def forward(self, bottom, top):
    top[1].data[...] = self._end_of_sequence
    top[2].data[...] = self._ignore_label
    # Score captions and generate training input and target output
    image_ids = []
    captions = []
    for n in range(self._batch_size):
      for b in range(self._beam_size):
        image_ids.append(int(bottom[0].data[n][0]))
        seq = bottom[1].data[n][0][b]
        captions.append(self._translate(seq))
        caption = seq[:len(captions[-1])].tolist()
        top[1].data[n*self._beam_size+b,1:min(self._sequence_length,len(caption)+1)] = \
            caption[:self._sequence_length-1] # input_sentence
        caption.append(self._end_of_sequence)
        top[2].data[n*self._beam_size+b,:min(self._sequence_length,len(caption))] = \
            caption[:self._sequence_length] # target_sentence
    raw_scores = np.array(self._cider.compute_scores(image_ids,captions))
    # Generate score output
    for n in range(self._batch_size):
      baseline = np.mean(raw_scores[n*self._beam_size:(n+1)*self._beam_size])
      for b in range(self._beam_size):
        score = raw_scores[n*self._beam_size+b]
        top[0].data[n*self._beam_size+b,:] = score - baseline
    top[3].data[0] = np.mean(raw_scores)

  def backward(self, top, propagate_down, bottom):
    """This layer does not propagate gradients."""
    pass    


class AlternateSelectionLayer(caffe.Layer):
  """
  Given two inputs, pick output alternately from each, starting with the first input.
  """

  def setup(self, bottom, top):
    if len(bottom) != 2:
      raise Exception("Inputs 2 bottom blobs - argmax and sampled inputs.")
    if len(top) != 1:
      raise Exception("Outputs 1 top blob - alternate selection.")
    assert bottom[0].data.shape == bottom[1].data.shape
      
  def reshape(self, bottom, top):
    top[0].reshape(*bottom[0].shape)

  def forward(self, bottom, top):
    t = np.empty((top[0].data.size))
    t[::2] = bottom[0].data.flatten()[::2]
    t[1::2] = bottom[1].data.flatten()[1::2]
    top[0].data[...] = t.reshape(top[0].data.shape)

  def backward(self, top, propagate_down, bottom):
    """This layer does not propagate gradients."""
    pass


class SCSTSamplingLayer(caffe.Layer):
  """
  Self-Critical Sequence Training (SCST) layer. Takes argmax and sampled captions and
  outputs weights for training.
  """

  def setup(self, bottom, top):
    if len(bottom) != 2:
      raise Exception("Inputs 2 bottom blobs - image_ids and captions.")
    if len(top) != 4:
      raise Exception("Outputs 4 top blobs - score_weights, target_sentence, mean_score, scores.")
    params = ast.literal_eval(self.param_str)
    self._end_of_sequence = params['end_of_sequence']
    self._ignore_label = params['ignore_label']
    # Load vocab
    self._vocab = []
    with open(params['vocab_path']) as vocab_file:
      for word in vocab_file:
        self._vocab.append(word.lower().strip())
    self._cider = CiderScorer(params['gt_caption_paths'], include_eos=True)
    
  def _translate(self, blob):
    # Results will be lower case, tokenized, without full stop
    # (to match reference tokenization)
    caption = [];
    for ix in blob:
      next_word = self._vocab[int(ix)]
      if next_word == '.':
        caption.append(next_word) # Include EOS
        break
      caption.append(next_word)
    return caption
      
  def reshape(self, bottom, top):
    self._batch_size = bottom[1].shape[0]
    self._sequence_length = bottom[1].shape[1]
    top[0].reshape(self._batch_size, self._sequence_length)
    top[1].reshape(self._batch_size, self._sequence_length)
    top[2].reshape(1)
    top[3].reshape(self._batch_size)

  def forward(self, bottom, top):
    top[0].data[...] = 0
    top[1].data[...] = self._ignore_label
    # Score captions and generate target output
    image_ids = []
    captions = []
    for n in range(self._batch_size):
      image_ids.append(int(bottom[0].data[n/2][0]))
      seq = bottom[1].data[n]
      captions.append(self._translate(seq))
      if n % 2 == 1: # Generate targets
        caption = seq[:len(captions[-1])].tolist()
        top[1].data[n,:min(self._sequence_length,len(caption))] = \
            caption[:self._sequence_length] # target_sentence
    raw_scores = self._cider.compute_scores(image_ids,captions)
    # Generate score weights
    for n in range(self._batch_size/2):
      baseline_score = raw_scores[n*2]
      sample_score = raw_scores[n*2+1]
      top[3].data[n*2] = baseline_score
      top[3].data[n*2+1] = sample_score
      if sample_score > 0:
        sample_score = math.log(sample_score)
      if baseline_score > 0:
        baseline_score = math.log(baseline_score)
      top[0].data[n*2+1] = max(0.0, sample_score - baseline_score)
    top[2].data[0] = np.mean(raw_scores[::2])

  def backward(self, top, propagate_down, bottom):
    """This layer does not propagate gradients."""
    pass    

