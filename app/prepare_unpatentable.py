import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm 
from nltk.tokenize import sent_tokenize
import numpy as np
import os

df_not_unpantable = pd.read_excel('./data/Unpatentable/ML08 - PTAB Case Data.xlsx', 'Not Unpatentable')
df_unpantable = pd.read_excel('./data/Unpatentable/ML08 - PTAB Case Data.xlsx', 'Unpatentable-Cancelled')
df = pd.concat([df_not_unpantable,df_unpantable])
df = df.dropna(axis=1,how='all')
df = df.dropna(subset=['Claim 1 of Patent'],axis=0)
df.to_excel("./data/Unpatentable/Final ML08 - PTAB Case Data.xlsx",index=False)

model = SentenceTransformer('all-MiniLM-L6-v2')



info_df = pd.DataFrame(columns=['ID','path'])
ids = []
paths = []
base_save_path = 'data Unpatentable Embeddings'
for index, row in tqdm(df.iterrows()):
    sents = sent_tokenize(row['Claim 1 of Patent'])
    embs = []
    for sent in sents: 
        emb = model.encode(sent)
        embs.append(emb)
    final_emb = np.mean(embs,axis=0)
    save_path_decomp = f'{base_save_path} {index}.npy'
    save_path = os.path.join(*save_path_decomp.split(' '))
    np.save(save_path, final_emb)
    ids.append(index)
    paths.append(save_path_decomp)
info_df['ID'] = ids
info_df['path'] = paths
info_df.to_csv(os.path.join('data','Unpatentable','info.csv'))
    