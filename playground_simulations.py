import requests
from groq import Groq
import pandas as pd
import os
import matplotlib.pyplot as plt

# carico il dataset dei libri come un dataframe pandas
data_set = pd.read_csv('/Users/IG45918/PycharmProjects/pythonProject/Project work/Datasets/trainingset_with_name.tsv', sep='\t',
                        header=None, names=['userID', 'bookID', 'rating', 'name'],
                        usecols=['userID', 'bookID', 'rating', 'name'])

# creo la chiave per l' api
apikey = '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'
# istanziare il client
client = Groq(api_key=apikey)

# quali modelli voglio interrogare?
models = ['LLaMa-3.3-70b-versatile', "LLaMa-3.1-70b-versatile", "LLaMa-3.2-3b-preview", "LLaMa-3.2-1b-preview",
          "LLaMa-3.1-8b-instant"]

# creo la directory dove archiviare le risposta
directory = '/Users/IG45918/PycharmProjects/pythonProject/Project work/Playground'

# creo dizionario per archiviare liste di recommendations
recommendations = {}

# creo dizionario per archiviare indici di Jaccard
data = {'user':[0],'model':[0],'prompt':[0],'jaccard':[0]}
df_jaccard = pd.DataFrame(data)

# circoscrivo il test ad un limitato numero di utenti
limit = 30
users = data_set['userID'].unique()
for userID in users[0:limit]:
    userID = int(userID)

    # quanti item, tra quelli disponibili passare come preferenze note dell' utente?
    train_set_size = 5
    train_set = data_set[data_set['userID'] == userID].iloc[0:train_set_size-1]
    test_set = data_set[data_set['userID'] == userID].iloc[5:10]
    train_set = train_set[['name','rating','bookID']].sort_values('rating',ascending=False)
    #print(f"The train set is:\n{train_set}")
    test_set = test_set[['name','rating','bookID']].sort_values('rating',ascending=False)
    #print(f"The test set is:\n{test_set}")
    reference_set = set(test_set['name'].tolist())

    # converto il df pandas con le preferenze dell' utente in una stringa da passare all' LLM
    user_pref = ', '.join(str(row['name']) for _,row in train_set.iterrows())

    # creo i 3 prompt
    recommended_items = 10-train_set_size
    prompt_CoT_0 = f"The user {userID} likes the following books: {user_pref}. Think step by step and then\
                provide me {recommended_items} ranked suggestions, based on these preferences: provide names only, nothing else."
    prompt_CoT_1 = f"Q) I like Sapiens by Harari and The selfish gene by Dawkins, provide me 1 suggestion, based on these preferences:\
        A) The books that you like are both scientific essays. The first one is an history book that offers a very broad range narrative of the 4 main revolutions\
        that humanity has been through so far: the cognitive revolution, the agricultural revolution, the industrial revolution and the digital revolution.\
        The second book explains how the behaviour of all the living species can be ultimately led back to their genes’ fight for survival. \
        Based on these preferences, I suggest you The naked ape by Desmond Morris, a book that describes how the mankind has evolved from apes,\
        combining biological history and life science from a scientific perspective. Q) You know that the user {userID} likes the following books: {user_pref}.\
        Provide me ranked {recommended_items} suggestions, based on these preferences: provide names only, nothing else."
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

    #invio richiesta
    for type_prompt,prompt in prompts.items():
        for my_model in models:
            response_to_list = []

            # creo un file .txt nella directory definita, in cui archiviare la risposta per prompt e per modello
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

                # estraggo la risposta come un oggetto di tipo stringa
                response = completion.choices[0].message.content

                # per il prompt generated knowledge genero il prompt definitivo
                if type_prompt == "prompt_GK":
                    #print(f"The knowledge generated is {completion.choices[0].message.content}")
                    prompt = f"Question) The user {userID} likes {user_pref}. Please provide {recommended_items} ranked suggestions, based on these preferences.\
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
                #print(completion.choices[0].message.content)

                # salvo la risposta in un file di testo
                with open(file,'w') as f:
                    f.write(response)
                # converto la risposta in una lista iterabile
                for i in range(0,recommended_items-1):
                    index_start = response.find(f"{i+1}.")
                    index_end = response.find(f"{i+2}.")
                    response_to_list.append(str(response[index_start+3:index_end-1]))
                response_to_list.append(str(response[index_end+3:]))

                #salvo la risposta come lista in un dizionario
                recommendations[f"user{userID}_{my_model}_{type_prompt}"] = [response_to_list]

                #salvo l' indice di Jaccard nel dataframe inizializzato prima
                row = pd.DataFrame({'user':[userID],'model':[my_model],'prompt':[type_prompt],'jaccard':[round(len(set(response_to_list) & reference_set)/len(set(response_to_list) | reference_set),2)]})
                df_jaccard = pd.concat([df_jaccard,row],ignore_index=True)
            except Exception as e:
                print("Errore durante la richiesta: ",e)

    #jaccard_index = dict(sorted(jaccard_index.items(), key = lambda item: item[1],reverse=True))
df_jaccard.drop(index=0,inplace=True)
print(df_jaccard.head())
#print(df_jaccard[df_jaccard['user']==2])

# interpreto i risultati con un barplot, raggruppando i dati per modello e poi per prompt
model_grouped = df_jaccard.groupby('model')['jaccard'].mean()
model_grouped.plot(kind='bar',color='blue',edgecolor='black')
plt.title('Jaccard index per model')
plt.xlabel('model')
plt.ylabel('J')
plt.tight_layout()
plt.show()
prompt_grouped = df_jaccard.groupby('prompt')['jaccard'].mean()
prompt_grouped.plot(kind='bar',color='red',edgecolor='black')
plt.title('Jaccard index per prompt')
plt.xlabel('prompt')
plt.ylabel('J')
plt.tight_layout()
plt.show()









                                                                                                                                                                      
                                                                                                                                                                                                                       
                                                                                                                                                                                                                       
                                                                                                                                                                                                                       
                                                                                                                                                                                                                       
                                                                                                                                                                                                                       
                                                                                                                                                                                                                       
                                                                                                                                                                                                                       
                                                                                                                                                                                                                       
