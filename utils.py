import pandas as pd
import numpy as np
from openai import OpenAI

class Undefined:
    #constructor method for ensuring that, when instanced, a new object of this class has its own inizialized data
    def __init__(self,data):
        self.data = data

def merge_ratings_movies(df_ratings,df_movies)
# left join between ratings and movies to preserve all the ratings rows
    df_ratings = pd.merge(df_ratings, df_movies, how='left', on='movieId')
    del df_ratings['movieId']
    return df_ratings
    
class Open_AI():
    # constructor method for ensuring that, when instanced, a new object of this class has its own model
    def __init__(self,model):
        self.model = model

    def question(self,message):
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
                                                
                                                                           
                                                                            
                                                            
        
