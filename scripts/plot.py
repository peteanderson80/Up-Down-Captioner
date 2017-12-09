#!/usr/bin/env python


# Plot automatic evaluation metrics on generated captions.
# First run evaluate.py to generate _avg_scores.json files.


import os
import json
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

SCORE_DIR = './scores/'
SUFFIX = '_avg_scores.json'
PLOT_DIR = './plots/'

EXCLUDED = [
  'lrcn_vgg_clip',
  'lrcn_vgg_decay',
  'lrcn_vgg_decay_drop',
  'lrcn_vgg_drop',
  'lrcn_vgg_att_full_context_joined',
  'lrcn_vgg_att_full_context_separate',
  'lrcn_vgg_att_full_context_joined_correct',
  'lrcn_vgg_att_full_context_separate_correct',
  'lrcn_resnet50_fixed',
  'lrcn_resnet50_fixed_adam',
  'lrcn_resnet50_fixed_adam_sentinel',
  'lrcn_vgg_att_EM'
]

def print_dataframe(df):
  table = pd.pivot_table(df, index=['iteration'], columns=['model'])
  print table


def load_dataframe(required=None,excluded=EXCLUDED):
  df = []
  for d in os.listdir(SCORE_DIR):
    for f in os.listdir(SCORE_DIR+d):
      if f.endswith(SUFFIX) and (not required or required in d) and (not excluded or d not in excluded):
        with open(SCORE_DIR + '/' + d + '/' + f) as data_file:
          try:
            data = json.load(data_file)
            df.append(data)
            df[-1]['model'] = d.split('/')[-1]
            df[-1]['iteration'] = int(f.split('_iter_')[1].split('_')[0])
          except Exception,e: 
            print str(e)
            print f
  return pd.DataFrame(df)

def table():
  df = []
  for d in os.listdir(SCORE_DIR):
    if d.startswith('resnet152'):
      for f in os.listdir(SCORE_DIR+d):
        if f.endswith(SUFFIX):
          with open(SCORE_DIR + '/' + d + '/' + f) as data_file:
            try:
              data = json.load(data_file)
              df.append(data)
              df[-1]['model'] = d.split('/')[-1] + '/' + f.split('iter_')[0]
              df[-1]['iteration'] = int(f.split('_iter_')[1].split('_')[0])
            except Exception,e: 
              print str(e)
              print f
  df = pd.DataFrame(df)
  df.to_excel('scores.xls')

def plot():
  df = load_dataframe()
  best_published = {
    'CIDEr' : 1.123,
    'METEOR' : 0.268,
    'ROUGE_L' : 0.559,
    'Bleu_1' : 0.773,
    'Bleu_2' : 0.609,
    'Bleu_3' : 0.461,
    'Bleu_4' : 0.344    
  }
  best_unpublished = {}
  if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)
  for var in df.columns.values:
    if var in ['iteration','model']:
      continue
    table = pd.pivot_table(df, values=var, index=['iteration'], columns=['model'])
    ax = table.plot(marker='o', colormap='jet')
    if var in best_published:
      line = plt.axhline(y=best_published[var], color='r')
      line.set_label('Best Published (Test Set)')
    if var in best_unpublished:
      line = plt.axhline(y=best_unpublished[var], color='r', linestyle='--')
      line.set_label('Best Unpublished (Test Set)')
    ax.legend(prop={'size':8}, loc='lower right')
    title = '%s' % (var)
    plt.title(title.replace('_',' '))
    plt.xlabel('Iteration')
    plt.savefig(PLOT_DIR + title + '.png')


if __name__ == "__main__":
  #plot()
  table()


