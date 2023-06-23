import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm 
import pickle as pkl
from nltk.tokenize import sent_tokenize
import numpy as np
import os
from datetime import datetime

df = pd.read_excel('./data/Gipson/Final-Gibson-Dunn-101-Cases.xlsx')

## Process the Gipson df
# df = df.fillna('')
# # for index, row in tqdm(df.iterrows()):
# #     if type(row['Date']) != datetime:
# #         print("Date: ",row['Date'])
# df['Date'] = df['Date'].apply(lambda x: x.strftime('%d/%m/%Y') if type(x) !=str else x)
# df.to_excel('./data/Gipson/Final-Gibson-Dunn-101-Cases.xlsx', index = False)


model = SentenceTransformer('all-MiniLM-L6-v2')
print(df.head())


info_df = pd.DataFrame(columns=['ID','path'])
ids = []
paths = []
base_save_path = os.path.join('data','Gipson','Embeddings')
for index, row in tqdm(df.iterrows()):
    sents = sent_tokenize(row['Holding'])
    embs = []
    for sent in sents: 
        emb = model.encode(sent)
        embs.append(emb)
    final_emb = np.mean(embs,axis=0)
    save_path = os.path.join(base_save_path,f'{index}.npy')
    np.save(save_path, final_emb)
    ids.append(index)
    paths.append(save_path)
info_df['ID'] = ids
info_df['path'] = paths
info_df.to_csv(os.path.join('data','Gipson','info.csv'))
    