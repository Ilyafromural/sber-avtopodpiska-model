import dill
import os

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Загружаем модель
model_filename = f'./model/' + sorted(os.listdir(f'./model/'))[-1]
with open(model_filename, 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    session_id: str
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):    
    prediction: float


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    import pandas as pd
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)

    return {        
        'prediction': y[0]
    }
