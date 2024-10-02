from fastapi import FastAPI
import uvicorn
from pandas import DataFrame

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.predict import predict

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/predict/title={title}&text={text}")
def read_item(title:str, text:str):
    # not rated and doesn't have a score yet, since it is a new review
    helpfulness = ['0/0']
    score = [0]
    return {"result": predict(DataFrame({'Title': [title], 'Helpfulness': helpfulness, 'Score': score, 'Text': [text]}))}

if __name__ == "__main__":
    uvicorn.run("main:app",port=8000, host='0.0.0.0')
