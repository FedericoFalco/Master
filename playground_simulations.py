import requests
from groq import Groq
import pandas as pd
import os
# carico il dataset dei libri come un dataframe pandas
data_set = pd.read_csv('/Users/IG45918/PycharmProjects/pythonProject/Project work/Datasets/trainingset_with_name.tsv', sep='\t',
                        header=None, names=['userID', 'bookID', 'rating', 'name'],
                        usecols=['userID', 'bookID', 'rating', 'name'])
# estraggo le sole preferenze del primo utente
userID = 1
train_set = data_set[data_set['userID'] == userID].iloc[0:4]
test_set = data_set[data_set['userID'] == userID].iloc[5:10]
train_set = train_set[['name','rating','bookID']].sort_values('rating',ascending=False)
user_pref = ', '.join(str(row['name']) for _,row in train_set.iterrows())
# creo i 3 prompt
prompt_CoT_0 = f"The user {userID} likes the following books: {user_pref}. Think step by step and then\
            provide me 5 ranked suggestions, based on these preferences: provide names only, nothing else."
prompt_CoT_1 = f"Q) I like Sapiens by Harari and The selfish gene by Dawkins, provide me 1 suggestion, based on these preferences:\
    A) The books that you like are both scientific essays. The first one is an history book that offers a very broad range narrative of the 4 main revolutions\
    that humanity has been through so far: the cognitive revolution, the agricultural revolution, the industrial revolution and the digital revolution.\
    The second book explains how the behaviour of all the living species can be ultimately led back to their genes’ fight for survival. \
    Based on these preferences, I suggest you The naked ape by Desmond Morris, a book that describes how the mankind has evolved from apes,\
    combining biological history and life science from a scientific perspective. Q) You know that the user {userID} likes the following books: {user_pref}.\
    Provide me ranked 5 suggestions, based on these preferences: provide names only, nothing else."
prompt_GK = f"Input) I like Sapiens by Harari, The elegant universe by Greene and The selfish gene by Dawkins.\
    Knowledge) These books are scientific essays based on the up-to-date discoveries in the fields of history, astrophysics and biology.\
    Unexpectedly they are fluid and easy to read, thanks to the writing skills of their author, who stick to evidence-based truths and broadly accepted thesis.\
    Input) The user {userID} likes {user_pref}.\
    Knowledge)"
prompts = {
    "prompt_CoT_0":prompt_CoT_0,
    "prompt_CoT_1":prompt_CoT_1,
    "prompt_GK":prompt_GK
}
#creo l' chiave per l' api
apikey = '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'
#istanziare il client
client = Groq(api_key = apikey)
#quali modelli voglio interrogare?
models = ['LLaMa-3.3-70b-versatile',"LLaMa-3.1-70b-versatile","LLaMa-3.2-3b-preview","LLaMa-3.2-1b-preview","LLaMa-3.1-8b-instant"]
#creo la directory dove archiviare le risposta
directory = '/Users/IG45918/PycharmProjects/pythonProject/Project work/Playground'
#invio richiesta
for type_prompt,prompt in prompts.items():
    for my_model in models:
        #creo un file .txt nella directory definita, in cui archiviare la risposta per prompt e per modello
        file = os.path.join(directory,f'user_{userID}_model_{my_model}_prompt_{type_prompt}.txt')
        try:
            completion = client.chat.completions.create(
                model=my_model,
                messages=[{"role":"user","content":prompt}],
                temperature=0,
                max_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )
            #estraggo la risposta come un oggetto di tipo stringa
            response = completion.choices[0].message.content
            #per il prompt generated knowledge genero il prompt definitivo
            if type_prompt == "prompt_GK":
                print(f"The knowledge generated is {completion.choices[0].message.content}")
                prompt = f"Question) The user {userID} likes {user_pref}. Please provide 5 ranked suggestions, based on these preferences.\
                            Provide names only, nothing else.\
                            Knowledge) {response}\
                            Answer) 'names only'"
                completion = client.chat.completions.create(
                    model=my_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=1024,
                    top_p=1,
                    stream=False,
                    stop=None,
                )
                response = completion.choices[0].message.content
            print(completion.choices[0].message,type(completion.choices[0].message))
            #salvo la risposta in un file di testo
            with open(file, 'w') as f:
                f.write(response)
        except Exception as e:
            print("Errore durante la richiesta: ",e)