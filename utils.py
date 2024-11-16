import pandas as pd
import numpy as np

class Undefined:
    #constructor method for ensuring that, when instanced, a new object of this class has its own inizialized data
    def __init__(self,data):
        self.data = data

def merge_ratings_movies(df_ratings,df_movies)
# left join between ratings and movies to preserve all the ratings rows
    df_ratings = pd.merge(df_ratings, df_movies, how='left', on='movieId')
    del df_ratings['movieId']
    return df_ratings
    
