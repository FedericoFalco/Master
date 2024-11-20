import pandas as pd
import numpy as np
from openai import OpenAI

class Undefined:
    #constructor method for ensuring that, when instanced, a new object of this class has its own inizialized data
    def __init__(self,data):
        self.data = data

    def book_read(self,user_ID):
        book_user = self.data[self.data['userID'] == user_ID]
        book_user = book_user[['name','rating','bookID']].sort_values('rating',ascending=0)
        book_user = book_user['name'].to_numpy()
        sentence = "The user {user_ID} likes the following books: {book_user}"
        return sentence

    def artists_listened(self,user_ID):
        artists_listened = self.data[self.data['userID'] == 'user_ID']
        artists_listened = artists_listened[['name','weight']].sort_values('rating',ascending=0)
        artists_listened = artists_listened['name'].to_numpy()
        sentence = "The user {user_ID} listens to the following artists: {artists_listened}"
        return sentence

    def movie_rated(self,user_ID):
        movie_rated = self.data[self.data['userID'] == 'user_ID']
        movie_rated = movie_rated[['movie','rating']]
        movie_rated = ", ".join(str(row['title']) + f{"int(row['rating'])} out of 5" for _,row in movie_rated.iterrows())
        sentence = "The user {user_ID} likes the following movies: {movie_rated}"
        return sentence

def merge_ratings_movies(df_ratings,df_movies)
# left join between ratings and movies to preserve all the ratings rows
    df_ratings = pd.merge(df_ratings, df_movies, how='left', on='movieID')
    del df_ratings['movieID']
    return df_ratings
    
class Open_AI:
    # constructor method for ensuring that, when instanced, a new object of this class has its own model
    def __init__(self,model):
        self.model = model

    def request(self,message):
        # calling the OpenAI api
        client = OpenAI(api_key = 'tbd')
        # setting the conversation
        answer = client.chat.completions.create(model=self.model, message=[{"role": "system", "content": "
                        "Given a user, as a Recommender System, please provide only the name of the top 50 recommendations."},
                        {"role": "user", "content": message}],
                                                temperature=0,
                                                max_tokens=500,
                                                top_p=1,
                                                frequence_penalty=0,
                                                presence_penalty=0)
        return answer
                                                
                                                                           
                                                                            
                                                            
        
