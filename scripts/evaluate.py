#!/usr/bin/python


# Run automatic evaluation metrics on generated captions.


import sys
import json
import os
import numpy as np
from plot import plot


COCO_ANN_PATH_VAL = 'data/coco/captions_val2014.json'
OUTPUTS_PATH = 'outputs/'

sys.path.append('./external/coco/PythonAPI/')
from pycocotools.coco import COCO

sys.path.append('./external/coco-caption')
from pycocoevalcap.eval import COCOEvalCap


class CaptionScorer():
  ''' Score captions on the COCO validation set. '''

  def __init__(self, gt_path = COCO_ANN_PATH_VAL):
    # Set up coco tools
    self.coco = COCO(gt_path)

  def score(self, json_path, save_results=True):
    if not os.path.exists(json_path):
      print "Not found: %s" % json_path
      return
    print "Evaluating captions in %s" % json_path
    generation_result = self.coco.loadRes(json_path)
    self.coco_evaluator = COCOEvalCap(self.coco, generation_result)
    # Set imageids to only those in the generation result
    self.coco_evaluator.params = {'image_id': generation_result.imgToAnns.keys()}
    self.coco_evaluator.evaluate()
    if save_results:
      self.in_path = json_path
      json_path = json_path.replace('outputs', 'scores')
      directory = '/'.join(json_path.split('/')[:-1])
      if not os.path.exists(directory):
        os.makedirs(directory)
      self.dump_individual_scores(json_path.replace('.json', '_scores.json'))
      self.dump_total_scores(json_path.replace('.json', '_avg_scores.json'))

  def dump_total_scores(self, scores_filename):
    print 'Dumping detailed scores to file: %s' % scores_filename
    s = dict(self.coco_evaluator.eval)
    # Add SPICE subcategories
    cats = ['Object', 'Attribute', 'Relation', 'Cardinality', 'Color', 'Size']
    for cat in cats:
      s[cat] = []   
    for img_id, evals in self.coco_evaluator.imgToEval.iteritems():
      for cat in cats:
        s[cat].append(evals['SPICE'][cat]['f'])
    for cat in cats:
      s[cat] = np.nanmean(np.array(s[cat]))
    # Save output
    with open(scores_filename, 'w') as json_file:
      json.dump(s, json_file, sort_keys=True, indent=2)    

  def dump_individual_scores(self, scores_filename):
    print 'Dumping individual scores to file: %s' % scores_filename
    with open(scores_filename, 'w') as json_file:
      json.dump(self.coco_evaluator.imgToEval, json_file, sort_keys=True, indent=2)

  def dump_total_scores_basic(self, scores_filename):
    print 'Dumping total scores to file: %s' % scores_filename
    with open(scores_filename, 'w') as json_file:
      json.dump(self.coco_evaluator.eval, json_file, sort_keys=True, indent=2)

def evaluate_range(model, start, stop, step):
  s = CaptionScorer()
  for it in range(start, stop, step):
    s.score(OUTPUTS_PATH + (model % it))
  plot()

def evaluate_model(model):
  s = CaptionScorer()
  s.score(OUTPUTS_PATH + model)
  plot()


