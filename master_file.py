import pandas as pd
import numpy as np
import * from utils

def chatGPT_recommender(model,dataset):
  # using the Open_AI function created to call the ChatGPT API
  if model == 'gpt-3.5-turbo-1106':
    recommender = Open_AI('gpt-3.5-turbo-1106')
  elif model == 'gpt-4-0125-preview':
    recommender = Open_AI('gpt-4-0125-preview')
  else:
    print('no valid model)
    exit()
    
  # retrieving data in a pandas df
  if dataset == 'facebook_book':
    train_set = pd.read_csv('../trainingset_with_name.tsv', sep="\t",
                            header=None, names=['userID', 'bookID', 'rating', 'name'],
                            usecols=['userID', 'bookID', 'rating', 'name'])
    # calling the class defined in the class_script
    trainset = Undefined(train_set)
  elif dataset == 'hetrec2011_lastfm_2k':
    train_set = pd.read_csv('../train_with_name.tsv', sep="\t",
                            header=None, names=['userID', 'artistID', 'weight', 'name', 'url', 'pictureURL'],
                            usecols=['userId', 'artistID', 'weight', 'name'])
    trainset = Undefined(train_set)
  elif dataset == 'ml_small_2018':
    # assume to having picked EXP 1, with only recommendations to be asked to ChatGPT
    ratings = pd.read_csv('../subset_train_230.tsv', sep="\t",
                            header=None, names=['userID', 'movieID', 'rating'])
    movies = pd.read_csv('../movies.csv', sep=',', 
                            header=None, names=['movieID','title','genre'],
                            usecols=['movieID','title'])
    train_set = merge_ratings_movies(ratings,movies)
    trainset = Undefined(train_set)
  else:
    print('No dataset found')
    exit()

# iterating over users included in the dataset
for user in trainset['userId'].unique():
  message = ''
  if dataset == 'facebook_book':
    message = train_set.book_read(user)
  elif dataset == 'hetrec2011_lastfm_2k':
    message = train_set.artists_listened(user)
  elif dataset == 'ml_small_2018':
    message = train_set.movie_rated(user)
    

    
