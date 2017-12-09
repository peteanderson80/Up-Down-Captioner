#!/usr/bin/python


''' This script depends on steps 1-3 of README.md
    Pre-process COCO train, val, test and 
    testserver sets (based on Karpathy splits).'''


import os
import sys
import re
import string
import math
from collections import defaultdict
import random
import json


COCO_ANN_PATH = './data/coco'
KARPATHY_SPLITS = './data/coco_splits/karpathy_%s_images.txt' # train,val,test

# 'train' refers to the Karpathy train split of 113k images.
# 'trainval' refers to the full 123k coco train + coco val set (for test server submission).
TRAIN_VOCAB_PATH = './data/coco_splits/train_vocab.txt'
TRAINVAL_VOCAB_PATH = './data/coco_splits/trainval_vocab.txt'
TRAIN_CAPTION_PATH = './data/coco_splits/train_captions.txt'
TRAINVAL_CAPTION_PATH = './data/coco_splits/trainval_captions.txt'

sys.path.append('./external/coco/PythonAPI/')
from pycocotools.coco import COCO

UNK_IDENTIFIER = '<unk>' # Used to identify unknown words
SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)') # Split on any non-alphanumeric character

  
def split_sentence(sentence):
  """ break sentence into a list of words and punctuation """
  toks = []
  for word in [s.strip().lower() for s in SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]:
    # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
    if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
      toks += list(word)
    else:
      toks.append(word)
  # Remove '.' from the end of the sentence - 
  # this is EOS token that will be populated by data layer
  if toks[-1] != '.':
    return toks
  return toks[:-1]

  
def line_to_stream(vocabulary, sentence):
  """ Convert a pre-processed list of dictionary words to a 
      list of incremented vocab indices """
  stream = []
  for word in sentence:
    word = word.strip()
    if word in vocabulary:
      stream.append(vocabulary[word])
    else:  # unknown word; append UNK
      stream.append(vocabulary[UNK_IDENTIFIER])
  return stream


def load_karpathy_splits(dataset='train'):
  imgIds = set()
  with open(KARPATHY_SPLITS % dataset) as data_file:
    for line in data_file:
      imgIds.add(int(line.split()[-1]))
  return imgIds


def load_caption_vocab(vocab_path=TRAIN_VOCAB_PATH):
  vocab = []
  print 'Loading vocab from %s' % vocab_path
  with open(vocab_path) as vocab_file:
    for word in vocab_file:
      vocab.append(word.strip())
  vocab_inverted = {}
  for i,word in enumerate(vocab):
    vocab_inverted[word] = i
  return vocab,vocab_inverted


def build_caption_vocab(out_path, datasets=['train'], base_vocab=None, min_vocab_count=5):
  print 'Building vocab from %s...' % datasets

  train_ids = set()
  for dataset in datasets:
    train_ids |= load_karpathy_splits(dataset=dataset)
  words_to_count = defaultdict(int)

  for dataset in ['train','val']: # coco sources
    annFile='%s/captions_%s2014.json' % (COCO_ANN_PATH, dataset)
    coco = COCO(annFile)
    # Count word frequencies
    for image_id,anns in coco.imgToAnns.iteritems():
      if image_id in train_ids:
        for ann in anns:
          caption_sequence = split_sentence(ann['caption'])
          for word in caption_sequence:
            words_to_count[word.strip()] += 1

  # Sort words by count, then alphabetically
  words_by_count = sorted(words_to_count.keys(), key=lambda w: (-words_to_count[w], w))
  print 'Initialized vocabulary with %d words; top 10 words:' % len(words_by_count)
  for word in words_by_count[:10]:
    print '\t%s (%d)' % (word, words_to_count[word])

  # Add words to vocabulary
  if base_vocab:
    vocabulary, vocabulary_inverted = load_caption_vocab(vocab_path=base_vocab)
  else:
    vocabulary = ['.', UNK_IDENTIFIER]
    vocabulary_inverted = {'.': 0, UNK_IDENTIFIER: 1}
  offset = len(vocabulary)
  for word in words_by_count:
    if words_to_count[word] < min_vocab_count:
      break
    if word in vocabulary_inverted:
      continue
    vocabulary.append(word)
    vocabulary_inverted[word] = offset
    offset += 1
  print 'Final vocabulary (restricted to words with counts of %d+) has %d words' % \
      (min_vocab_count, len(vocabulary_inverted))

  # Save vocab
  print 'Dumping vocabulary to file: %s' % out_path
  with open(out_path, 'wb') as vocab_file:
    for word in vocabulary:
      vocab_file.write('%s\n' % word.encode('utf-8'))
  print 'Done.'


def preprocess_coco(out_path, datasets=['train'], base_vocab=TRAIN_VOCAB_PATH):
  """ Generate sequence data from the raw coco captions. """

  # Load vocab
  vocab, vocab_inverted = load_caption_vocab(vocab_path=base_vocab)

  # Build training set
  train_ids = set()
  for dataset in datasets:
    print 'Processing karpathy split: %s' % (dataset)
    train_ids |= load_karpathy_splits(dataset=dataset)

  sequences = []    
  for coco_dataset in ['train','val']:
    annFile='%s/captions_%s2014.json' % (COCO_ANN_PATH, coco_dataset)
    coco = COCO(annFile)
    for image_id,anns in coco.imgToAnns.iteritems():
      if image_id in train_ids:
        image_info = coco.imgs[image_id]
        image_path = '%s/%s' % (image_info['file_name'].split('_')[1], image_info['file_name'])
        for ann in anns:
          caption_sequence = split_sentence(ann['caption'])
          sequences.append((image_path, caption_sequence))

  # Randomize and save training sequences - this should be repeatable
  random.seed(1)
  random.shuffle(sequences)
  print 'Dumping training sequences to file: %s' % out_path
  with open(out_path, 'w') as train_file:
    for image_path, caption_sequence in sequences:
      train_file.write('%s | %s\n' % (image_path, \
          ' '.join([str(i) for i in line_to_stream(vocab_inverted,caption_sequence)])))


if __name__ == "__main__":
  build_caption_vocab(TRAIN_VOCAB_PATH, datasets=['train'])
  build_caption_vocab(TRAINVAL_VOCAB_PATH, datasets=['train', 'val', 'test'], base_vocab=TRAIN_VOCAB_PATH)
  preprocess_coco(TRAIN_CAPTION_PATH, datasets=['train'], base_vocab=TRAIN_VOCAB_PATH)
  preprocess_coco(TRAINVAL_CAPTION_PATH, datasets=['train', 'val', 'test'], base_vocab=TRAINVAL_VOCAB_PATH)


