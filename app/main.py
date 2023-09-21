import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import onnxruntime
from fastapi import FastAPI
import uvicorn
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from pathlib import Path
from pydantic import BaseModel
import scipy
from scipy import spatial
from nltk.tokenize import sent_tokenize
def get_vendor(sent):
    vendors = ['apple','samsung','cisco','sony','google','microsoft','facebook','amazon','rovi']
    sent = sent.lower()
    for v in vendors:
        if v in sent:
            return v
    return 'other'
def get_cat(sent):
    categories = {'electronics':['electronics'],'communications':['communications','networks','wireless','networks']}
    sent = sent.lower()
    for key in categories:
        for c in categories[key]:
            if c in sent:
                return key
    return 'other'
def convert_from_range_to_another(OldValue,OldMax,OldMin,NewMax,NewMin):
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)  
    NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    return NewValue
base_path=Path(__file__).resolve(strict=True).parent
base_events_meta_path = os.path.join(base_path,'meta_data','events_meta')
base_unpatentable_meta_path = os.path.join(base_path,'meta_data','unpatentable_meta')

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')
calc_cos_sem = lambda x,y: 1 - spatial.distance.cosine(x, y)

@app.get("/")
def root():
    return {"message": "Hello , this is Eligible claims classifier app ,write (/docs) in the URL to go to app utilities"}


 
class Item(BaseModel):
    representative_claim_input: str
    judge: str  = None
    court:str  = None
class UnpatentableItem(BaseModel):
    claim: str
    claim_name: str  = None
class ChooseImpSentsBody(BaseModel):
    spec:str
    claim:str
    n_sents:int=3
class GetGibsonRelevantSentsBody(BaseModel):
    request:str
    n_sents:int=5
class GetUnpatentableRelevantSentsBody(BaseModel):
    request:str
    n_sents:int=15
def get_top_n_embeddings(request,data_path,n):
    request_emb = model.encode(request)
    info_df = pd.read_csv(os.path.join(data_path,'info.csv'))
    scores = np.array([])
    indeces = np.array([])
    for  _, row in info_df.iterrows():
        emb = np.load(os.path.join(base_path,*row['path'].split(' ')))
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

@app.post("/get_unpatentable_relevant_sents")
async def get_gibson_relevant_sents(get_unpatentable_relevant_sents_body:GetUnpatentableRelevantSentsBody):
    request = get_unpatentable_relevant_sents_body.request
    n_sents = get_unpatentable_relevant_sents_body.n_sents
    gipson_path = os.path.join(base_path,'data','Unpatentable')
    indeces,scores = get_top_n_embeddings(request,gipson_path,n_sents)
    df = pd.read_excel(os.path.join(gipson_path,'Final ML08 - PTAB Case Data.xlsx'))
    sub_df = df.loc[indeces,:].to_dict('list')
    
    response = {'status':200,'Claim 1 of Patent':sub_df['Claim 1 of Patent'],'Case Name':sub_df['Case Name'],'Case Number':sub_df['Case Number'],'US Patent Number':sub_df['US Patent Number'],'Determination':sub_df['Determination'],'score':list(scores)}
    print(response)
    return response      

@app.post("/is_unpatentable")
async def is_unpatentable(request_data:UnpatentableItem):
    meta_data_path = os.path.join(base_path,'meta_data','unpatentable_meta')
    vendors_pickle_path = os.path.join(meta_data_path,'vendors.pkl')
    cats_pickle_path = os.path.join(meta_data_path,'cats.pkl')
    
    claim = request_data.claim
    claim_name = request_data.claim_name
    vendor = get_vendor(claim_name)
    category = get_cat(claim_name)
    if claim is None:
        return {"status":400,"Error":"Please provide a claim"}
    
    tokenizers_files_path =str(base_path)+"/tokenizer_files"
    model_path = str(base_path)+"/0.69-0.69-all.quant.onnx"
    tokenizer = AutoTokenizer.from_pretrained(tokenizers_files_path)
    inputs = tokenizer(
        claim,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="np",
    )
    
    inputs["input_ids"]=inputs["input_ids"].astype(np.int64)
    inputs["token_type_ids"]=inputs["token_type_ids"].astype(np.int64)
    inputs["attention_mask"]=inputs["attention_mask"].astype(np.int64)

    session = onnxruntime.InferenceSession(model_path)
    onnx_preds = session.run(None, dict(inputs))[0]
    preds_prob = scipy.special.softmax(onnx_preds,axis=1)
    
    with open(vendors_pickle_path, 'rb') as fp:
            vend_code_dict = pickle.load(fp)
    with open(cats_pickle_path, 'rb') as fp:
        cat_code_dict = pickle.load(fp)
    vendor_exists = vendor is not None and vendor in vend_code_dict
    cat_exists = category is not None and category in cat_code_dict
    if vendor_exists and cat_exists:
        xgb_model_path = os.path.join(meta_data_path,'xgb_unpantable.pickle')
        xgb_model = pickle.load(open(xgb_model_path, "rb"))
                
        preds = [p[0] for p in preds_prob]
        few_shot = [item.item() for item in preds]
        
        df = pd.DataFrame({'vendors_enc':vendor,'cat_enc':category,'FewShot':few_shot})
        df['vendors_enc'] = df['vendors_enc'].map(vend_code_dict)
        df['cat_enc'] = df['cat_enc'].map(cat_code_dict)
        
        pred_prob  = xgb_model.predict_proba(df)
        pred_prob = [p[0] for p in pred_prob]
        # preds = (np.array(pred_prob)<0.45).astype(int)
        confidence = []
        result = []
        for prob in pred_prob:
            if prob < 0.45:
                prob = 1-prob
                old_min = 0.55
                result.append(0)
            else:
                old_min = 0.45
                result.append(1)
            prob = convert_from_range_to_another(prob,1.0,old_min,1.0,0.5)
            confidence.append(prob)                
    else:
        result = np.argmax(preds_prob,axis=1)
        confidence = np.max(preds_prob,axis=1)
        result = result.tolist()
        confidence = confidence.tolist()
    confidence = np.round(confidence,4).tolist()[0]
    result = result[0]
    return {"status":200,"result":result,'confidence':confidence}

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
    with open(str(base_events_meta_path)+"/judge_conf.pkl", 'rb') as fp:
        judge_conf = pickle.load(fp)

    with open(str(base_events_meta_path)+"/eligible_per_judge.pkl", 'rb') as fp:
        eligible_per_judge = pickle.load(fp)

    with open(str(base_events_meta_path)+"/court_conf.pkl", 'rb') as fp:
        court_conf = pickle.load(fp)
    
    with open(str(base_events_meta_path)+"/eligible_per_court.pkl", 'rb') as fp:
        eligible_per_court = pickle.load(fp)

    judges=list(judge_conf.keys())
    
    courts=list(court_conf.keys())
    
    
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
            xgb_model.load_model(str(base_events_meta_path)+"/xgb_model_quantized_onnx_probabilities.json")  
            model_prediction=xgb_model.predict(model_input)

            return {"status":200,"result":model_prediction.item(),"confidence":round(xgb_model.predict_proba(model_input)[0].max().item(),6)}

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

if __name__ == "__main__":
    uvicorn.run(app,port=80)
    

