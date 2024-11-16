import pandas as pd
import numpy as np
import Undefined from utils

def get_dataset(dataset):
  # retrieving data in a Pandas df
  if dataset == 'facebook_book':
    train_set = pd.read_csv('../trainingset_with_name.tsv', sep="\t",
                            header=None, names=['userId', 'bookId', 'rating', 'name'],
                            usecols=['userId', 'bookId', 'rating', 'name'])
    # calling the class defined in the class_script
    trainset = Undefined(train_set)
  elif dataset == 'hetrec2011_lastfm_2k':
    train_set = pd.read_csv('../train_with_name.tsv', sep="\t",
                            header=None, names=['userId', 'artistId', 'weight', 'name', 'url', 'pictureURL'],
                            usecols=['userId', 'artistId', 'weight', 'name'])
    trainset = Undefined(train_set)
  elif dataset == 'ml_small_2018':
    # assume to having picked EXP 1, with only recommendations to be asked to ChatGPT
    ratings = pd.read_csv('../subset_train_230.tsv', sep="\t",
                            header=None, names=['userId', 'movieId', 'rating'])
    movies = pd.read_csv('../movies.csv', sep=',', 
                            header=None, names=['movieId','title','genre'],
                            usecols=['movieId','title'])
    train_set = merge_ratings_movies(ratings,movies)
    
