
from fastapi import FastAPI
import uvicorn

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

app = FastAPI()



@app.get("/")
def root():
    return {"message": "Hello , this is Eligible claims classifier app ,write (/docs) in the URL to go to app utilities"}


 
class Item(BaseModel):
    representative_claim_input: str
    judge: str | None = None
    court:str | None = None
    

@app.post("/predict")
async def inference(item: Item):    
    
    representative_claim=item.representative_claim_input.strip()
    court=item.court
    judge=item.judge
    base_path=Path(__file__).resolve(strict=True).parent

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

