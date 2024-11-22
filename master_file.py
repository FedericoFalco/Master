import pandas as pd
import numpy as np
import utils
import openai
from utils import Undefined, Open_AI

def chatGPT_recommender(model, dataset):
    recommender = None
    # using the Open_AI function created to call the ChatGPT API
    if model == 'gpt-3.5-turbo-instruct-0914':
        recommender = Open_AI('gpt-3.5-turbo-instruct-0914')
    elif model == 'gpt-4-0125-preview':
        recommender = Open_AI('gpt-4-0125-preview')
    else:
        print('no valid model')
        exit()
        # retrieving data in a pandas df
    if dataset == 'facebook_book':
        train_set = pd.read_csv('/Users/IG45918/PycharmProjects/pythonProject/Project work/trainingset_with_name.tsv', sep='\t',
                        header=None, names=['userID', 'bookID', 'rating', 'name'],
                        usecols=['userID', 'bookID', 'rating', 'name'])
        # calling the class defined in the class_script
        trainset = utils.Undefined(train_set)
    elif dataset == 'hetrec2011_lastfm_2k':
        train_set = pd.read_csv('/Users/IG45918/PycharmProjects/pythonProject/Project work/train_with_name.tsv', sep="\t",
                                header=None, names=['userID', 'artistID', 'weight', 'name', 'url', 'pictureURL'],
                                usecols=['userId', 'artistID', 'weight', 'name'])
        trainset = utils.Undefined(train_set)
    elif dataset == 'ml_small_2018':
        # assume to having picked EXP 1, with only recommendations to be asked to ChatGPT
        ratings = pd.read_csv('/Users/IG45918/PycharmProjects/pythonProject/Project work/subset_train_200.tsv', sep="\t",
                              header=None, names=['userID', 'movieID', 'rating'])
        movies = pd.read_csv('/Users/IG45918/PycharmProjects/pythonProject/movies.csv', sep=',',
                             header=None, names=['movieID', 'title', 'genre'],
                             usecols=['movieID', 'title'])
        train_set = utils.merge_ratings_movies(ratings, movies)
        trainset = utils.Undefined(train_set)
    else:
        print('No dataset found')

    # iterating over users included in the dataset and generating the message to pass to ChatGPT
    for user in train_set['userID'].unique():
        message = ''
        if dataset == 'facebook_book':
            message = trainset.book_read(user)
        elif dataset == 'hetrec2011_lastfm_2k':
            message = trainset.artists_listened(user)
        elif dataset == 'ml_small_2018':
            message = trainset.movie_rated(user)


    # passing the message to ChatGPT, by applying the pre-defined method 'request' on the object of class 'recommender'
    response = recommender.request(message)
    return response

# calling the function chatGPT_recommender
model = 'gpt-3.5-turbo-instruct-0914'
dataset = 'facebook_book'
chatGPT_recommender(model=model,dataset=dataset)
    
