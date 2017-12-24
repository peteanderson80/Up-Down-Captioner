#!/usr/bin/env python


# Plot automatic evaluation metrics on generated captions.
# First run evaluate.py to generate _avg_scores.json files.


import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCORE_DIR = './scores/'
SUFFIX = '_avg_scores.json'
PLOT_DIR = './plots/'


def print_dataframe(df):
  table = pd.pivot_table(df, index=['iteration'], columns=['model'])
  print table


def load_dataframe(required=None,excluded=None):
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


def plot():
  df = load_dataframe()
  if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)
  for var in df.columns.values:
    if var in ['iteration','model']:
      continue
    table = pd.pivot_table(df, values=var, index=['iteration'], columns=['model'])
    ax = table.plot(marker='o', colormap='jet')
    ax.legend(prop={'size':8}, loc='lower right')
    title = '%s' % (var)
    plt.title(title.replace('_',' '))
    plt.xlabel('Iteration')
    plt.savefig(PLOT_DIR + title + '.png')



