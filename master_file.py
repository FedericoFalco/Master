import pandas as pd
import numpy as np
import Undefined from class_script

def get_dataset(dataset):
  # retrieving data in a Pandas df
  if dataset == 'facebook_book':
    train_set = pd.read_csv('../data/dataset/facebook_book/trainingset_with_name.tsv', sep="\t",
                            header=None, names=['userId', 'bookId', 'rating', 'name'],
                            usecols=['userId', 'bookId', 'rating', 'name'])
    # calling the class defined in the class_script
    trainset = Undefined(train_set)
  elif dataset == 'hetrec2011_lastfm_2k':
    train_set = pd.read_csv('../data/dataset/hetrec2011_lastfm_2k/splitting/0/train_with_name.tsv', sep="\t",
                            header=None, names=['userId', 'artistId', 'weight', 'name', 'url', 'pictureURL'],
                            usecols=['userId', 'artistId', 'weight', 'name'])
    trainset = Undefined(train_set)
  elif dataset == 'ml_small_2018':
    # assume to having picked EXP 1, with only recommendations to be asked to ChatGPT
    ratings = pd.read_csv('../data/dataset/ml_small_2018/splitting/0/subset_train_230.tsv'
    train_set = pd.read_csv('../data
    
