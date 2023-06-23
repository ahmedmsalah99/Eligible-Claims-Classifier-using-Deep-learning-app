
from fastapi import FastAPI
import uvicorn
import os
import pickle
import pandas as pd

import xgboost as xgb
import json
import onnxruntime
from transformers import AutoTokenizer
import numpy as np
import scipy

from pathlib import Path
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from scipy import spatial
from nltk.tokenize import sent_tokenize

base_path=Path(__file__).resolve(strict=True).parent
app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')
calc_cos_sem = lambda x,y: 1 - spatial.distance.cosine(x, y)

@app.get("/")
def root():
    return {"message": "Hello , this is Eligible claims classifier app ,write (/docs) in the URL to go to app utilities"}


 
class Item(BaseModel):
    representative_claim_input: str
    judge: str | None = None
    court:str | None = None
class ChooseImpSentsBody(BaseModel):
    spec:str
    claim:str
    n_sents:int|None=3
class GetGibsonRelevantSentsBody(BaseModel):
    request:str
    n_sents:int|None=5
def get_top_n_embeddings(request,data_path,n):
    request_emb = model.encode(request)
    embeddings_path = os.path.join(data_path,'Embeddings')
    info_df = pd.read_csv(os.path.join(data_path,'info.csv'))
    scores = np.array([])
    indeces = np.array([])
    for  index, row in info_df.iterrows():
        emb = np.load(os.path.join(base_path,row['path']))
        score = calc_cos_sem(emb,request_emb)
        scores = np.append(scores,score)
        indeces = np.append(indeces,row['ID'])
    sort_indeces = np.argsort(scores)[::-1]
    indeces = indeces[sort_indeces]
    scores = scores[sort_indeces]
    return indeces[:n],scores[:n]

@app.post("/get_gibson_relevant_sents")
async def get_gibson_relevant_sents(get_gibson_relevant_sents_body:GetGibsonRelevantSentsBody):
    request = get_gibson_relevant_sents_body.request
    n_sents = get_gibson_relevant_sents_body.n_sents
    gipson_path = os.path.join(base_path,'data','Gipson')
    indeces,scores = get_top_n_embeddings(request,gipson_path,n_sents)
    df = pd.read_excel(os.path.join(gipson_path,'Final-Gibson-Dunn-101-Cases.xlsx'))
    sub_df = df.loc[indeces,:].to_dict('list')
    
    response = {'status':200,'holding':sub_df['Holding'],'case':sub_df['Case'],'date':sub_df['Date'],'elegibility':sub_df['Eligibility'],'score':list(scores)}
    print(response)
    return response
        

    

@app.post("/choose_imp_sents")
async def choose_imp_sents(choose_imp_sents_body:ChooseImpSentsBody):
    claim = choose_imp_sents_body.claim
    spec = choose_imp_sents_body.spec
    n_sents = choose_imp_sents_body.n_sents
    
    sent_bef = ''
    sent_after = ''
    sentences = sent_tokenize(spec)
    sentence_embeddings = model.encode(sentences)
    claim_embedding = model.encode(claim)

    scores = [calc_cos_sem(emb,claim_embedding) for emb in sentence_embeddings]
    indeces = np.argsort(scores)
    sorted_sentences = list(np.array(sentences)[indeces][::-1])

    for i in range(1,len(scores)-1):
        scores[i] = np.mean(scores[i-1:i+1])

    indeces = np.argsort(scores)
    if len(indeces)<1:
        return {"status":200,"relevant_part":"","sorted_sentences":[]}
    most_relevant_sent_idx = indeces[-1]
    most_relevant_sent = sentences[most_relevant_sent_idx]


    if most_relevant_sent_idx-1>-1:
        idx = most_relevant_sent_idx-1
        sent_bef = sentences[idx]

    if most_relevant_sent_idx+1<len(scores):
        idx = most_relevant_sent_idx+1
        sent_after = sentences[idx]
    sents = [sent for sent in [sent_bef,most_relevant_sent,sent_after] if sent != ""]
    relevant_part = "".join(sents)
    return {"status":200,"relevant_part":relevant_part,"sorted_sentences":sorted_sentences[:n_sents]}

@app.post("/predict")
async def inference(item: Item):    
    
    representative_claim=item.representative_claim_input.strip()
    court=item.court
    judge=item.judge
    

    tokenizers_files_path =str(base_path)+"/tokenizer_files"
    output_path = str(base_path)+"/quantized_setfitonnx_model.onnx"
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizers_files_path)
    inputs = tokenizer(
        representative_claim,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="np",
    )
    
    inputs["input_ids"]=inputs["input_ids"].astype(np.int64)
    inputs["token_type_ids"]=inputs["token_type_ids"].astype(np.int64)
    inputs["attention_mask"]=inputs["attention_mask"].astype(np.int64)

    session = onnxruntime.InferenceSession(output_path)


    # load dictionary
    with open(str(base_path)+"/judge_conf.pkl", 'rb') as fp:
        judge_conf = pickle.load(fp)

    with open(str(base_path)+"/eligible_per_judge.pkl", 'rb') as fp:
        eligible_per_judge = pickle.load(fp)

    with open(str(base_path)+"/court_conf.pkl", 'rb') as fp:
        court_conf = pickle.load(fp)
    
    with open(str(base_path)+"/eligible_per_court.pkl", 'rb') as fp:
        eligible_per_court = pickle.load(fp)

    judges=list(judge_conf.keys())
    
    courts=list(court_conf.keys())
    
    
    #np.logical_not().astype(int)
    if judge==None or court==None:
        onnx_preds = session.run(None, dict(inputs))[0]
        softmax=scipy.special.softmax(onnx_preds, axis=1)
        return {"status":200,"result": abs(1-np.argmax(softmax).item()),"confidence":round(np.max(softmax).item(),6)}
    else:
        # can load dictionary here
        judge=judge.strip()
        court=court.strip()
        if judge not in judges or court not in courts:
            onnx_preds = session.run(None, dict(inputs))[0]
            softmax=scipy.special.softmax(onnx_preds, axis=1)
            return {"status":200,"result": abs(1-np.argmax(softmax).item()),"confidence":round(np.max(softmax).item(),6)}
        else:

            prob_name = session.get_outputs()[0].name
            onnx_preds = session.run([prob_name], dict(inputs))[0]
        
            model_input=pd.DataFrame([[judge_conf[judge],
                            court_conf[court],
                            eligible_per_judge[judge],
                            eligible_per_court[court],
                            abs(1-scipy.special.softmax(onnx_preds, axis=1)[0][0])]
                            ],
                            columns=['judge_conf', 'court_conf', 'eligible_per_judge', 'eligible_per_court', 'FewShot'])        
        
        
            xgb_model = xgb.XGBClassifier(max_depth = 3, learning_rate=0.05,n_estimators=100,verposity=3,objective="binary:logistic", random_state=42)
            xgb_model.load_model(str(base_path)+"/xgb_model_quantized_onnx_probabilities.json")  
            model_prediction=xgb_model.predict(model_input)

            return {"status":200,"result":model_prediction.item(),"confidence":round(xgb_model.predict_proba(model_input)[0].max().item(),6)}


# if __name__ == "__main__":
#     uvicorn.run(app)
    

