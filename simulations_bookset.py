from functions_bookset import simulation
import pandas as pd
from groq import Groq
import numpy as np
import time
import matplotlib as plt

# carico il dataset dei libri come un dataframe pandas
data_set = pd.read_csv('/Users/IG45918/PycharmProjects/pythonProject/Project work/Datasets/trainingset_with_name.tsv',
                       sep='\t',
                       header=None, names=['userID', 'bookID', 'rating', 'name'],
                       usecols=['userID', 'bookID', 'rating', 'name'])

# estraggo gli utenti che abbiano almeno 20 item come preferenze
grouped = data_set.groupby(by='userID').count() #.sort_values(by='bookID', ascending=False)
userID_20 = grouped[grouped['bookID'] >= 20]
data_set = data_set[data_set['userID'].isin(userID_20.index)]

# inserisco la chiave api
apikey = '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'

# istanziare il client
client = Groq(api_key=apikey)

# quali modelli voglio interrogare?
models = ['LLaMa-3.3-70b-versatile', "LLaMa-3.2-3b-preview", "LLaMa-3.2-1b-preview",
          "LLaMa-3.1-8b-instant"]

# creo il nome della directory dove archiviare le risposta
directory = '/Users/IG45918/PycharmProjects/pythonProject/Project work/Playground_books_txts'

# estraggo l' elenco di utenti
users = data_set['userID'].unique()

'''
# cerco il valore di train_ratio che massimizzi la metrica di Jaccard -> 60%
train_vec = np.linspace(0.2,0.8,4)
data = {'train_ratio': [0], 'jaccard': [0]}
jaccard_vec = pd.DataFrame(data)

for i in train_vec:
    time.sleep(600)
    df_output = simulation(data_set=data_set ,models_list=models,client=client,users_list=users,num_of_users=10,train_ratio=i,
                            recomm_to_test=1, directory=directory)
    row = pd.DataFrame({'train_ratio': [i],'jaccard': max(df_output['jaccard'])})
    jaccard_vec = pd.concat([row,jaccard_vec])
print(jaccard_vec)

# analizzo i risultati con un barplot, raggruppando i dati per modello e poi per prompt
model_grouped = jaccard_vec.groupby('model')['jaccard'].mean()
model_grouped.plot(kind='bar', color='blue', edgecolor='black')
plt.title('Jaccard index per model')
plt.xlabel('model')
plt.ylabel('J')
plt.tight_layout()
plt.show()
prompt_grouped = jaccard_vec.groupby('prompt')['jaccard'].mean()
prompt_grouped.plot(kind='bar', color='red', edgecolor='black')
plt.title('Jaccard index per prompt')
plt.xlabel('prompt')
plt.ylabel('J')
plt.tight_layout()
plt.show()
'''

# fisso il train ratio al 60% e salvo in un excel i risultati delle simulazioni
df_output = simulation(data_set=data_set, models_list=models, client=client, users_list=users, num_of_users=len(users),
                        train_ratio=0.6, recomm_to_test=1, directory=directory)

# mostro tutte le righe e colonne
pd.set_option('display.max_rows', None)  # Mostra tutte le righe
pd.set_option('display.max_columns', None)  # Mostra tutte le colonne

print(df_output)
df_output.to_excel('simulations_books_2.xlsx')

# raggruppo i risultati per prompt e per modello
results = df_output.drop(['user','train_ratio','test_set','recommendations'], axis=1).groupby(['model','prompt']).mean()
print(results)
results.to_excel('results_books_2.xlsx')
