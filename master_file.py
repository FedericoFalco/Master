import pandas as pd
import numpy as np
import utils
import openai
from utils import Undefined, Open_AI
import os

def chatGPT_recommender(model,dataset,directory):
    recommender = None
    # using the Open_AI function created to call the ChatGPT API
    if model == 'gpt-3.5-turbo-1106':
        recommender = Open_AI('gpt-3.5-turbo-1106')
    elif model == 'gpt-4-0125-preview':
        recommender = Open_AI('gpt-4-0125-preview')
    else:
        print('no valid model')
        exit()
        # retrieving data in a pandas df
    if dataset == 'facebook_book':
        train_set = pd.read_csv('/Users/IG45918/PycharmProjects/pythonProject/Project work/Datasets/trainingset_with_name.tsv', sep='\t',
                        header=None, names=['userID', 'bookID', 'rating', 'name'],
                        usecols=['userID', 'bookID', 'rating', 'name'])
        # calling the class defined in the class_script
        trainset = utils.Undefined(train_set)
    elif dataset == 'hetrec2011_lastfm_2k':
        train_set = pd.read_csv('/Users/IG45918/PycharmProjects/pythonProject/Project work/Datasets/train_with_name.tsv', sep="\t",
                                header=None, names=['userID', 'artistID', 'weight', 'name', 'url', 'pictureURL'],
                                usecols=['userId', 'artistID', 'weight', 'name'])
        trainset = utils.Undefined(train_set)
    elif dataset == 'ml_small_2018':
        # assume to having picked EXP 1, with only recommendations to be asked to ChatGPT
        ratings = pd.read_csv('/Users/IG45918/PycharmProjects/pythonProject/Project work/Datasets/subset_train_200.tsv', sep="\t",
                              header=None, names=['userID', 'movieID', 'rating'])
        movies = pd.read_csv('/Users/IG45918/PycharmProjects/pythonProject/movies.csv', sep=',',
                             header=None, names=['movieID', 'title', 'genre'],
                             usecols=['movieID', 'title'])
        train_set = utils.merge_ratings_movies(ratings, movies)
        trainset = utils.Undefined(train_set)
    else:
        print('No dataset found')
        
    # checking if the users have already been considered
    last_user_checkpoint = None
    max_id = -1
    highest_user_file = None
    # 'os.scandir' gives a list of the files and sub-directories included in the directory given in input
    with os.scandir(checkpoint_dir) as files:
        for file in files:
            if file.name.startswith("user_") and file.name.endswith("_checkpoint.txt"):
                user_id = int(file.name[len("user_"):-len("_checkpoint.txt")])
                if user_id > max_id:
                    max_id = user_id
                    highest_user_file = file.path

    # iterating over users included in the dataset and generating the message to pass to ChatGPT
    for user in train_set['userID'].unique():
        if last_user_checkpoint is not None and user <= max_id:
            continue
        message = ''
        if dataset == 'facebook_book':
            message = trainset.book_read(user)
        elif dataset == 'hetrec2011_lastfm_2k':
            message = trainset.artists_listened(user)
        elif dataset == 'ml_small_2018':
            message = trainset.movie_rated(user)
        print(message)

        # passing the message to ChatGPT, by applying the pre-defined method 'request' on the object of class 'recommender'
        response = recommender.request(message)
        # creating a text file with the library 'os' to store the response
        file = os.path.join(directory, f'user_{user}_checkpoint.txt')
        # writing the first of the ChatGPT responses into the file
        with open(file,'w') as f:
            f.write(response.choices[0].message.content)

# calling the function chatGPT_recommender
model = 'gpt-3.5-turbo-1106'
dataset = 'facebook_book'
base_dir = '/Users/IG45918/PycharmProjects/pythonProject/Project work'
checkpoint_dir = os.path.join(base_dir,dataset,model,'TopNRec')

if os.path.exists(checkpoint_dir):
    chatGPT_recommender(model=model,dataset=dataset,directory=checkpoint_dir)
else:
    print(f"The directory {checkpoint_dir} does not exist")
