#pip install fastapi
#pip install uvicorn

from joblib import load
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris

iris = load_iris()

#Chargement du model
load_model=load("model.joblib")

#Création d'une nouvelle instance appli

app=FastAPI()

#Définir un objet pour réaliser des requetes

class request_body(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width :float
    
@app.post("/predict")

def predict(data : request_body) :
    #NOUVELLE DONNEES POUR LA PREDICTION
    new_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_width,
        data.petal_length
    ]]
    #PREDICTION
    class_idx= load_model.predict(new_data)[0]
    # Nous retournons le nom de l'espèce iris
    return {'class' : iris.target_names[class_idx]}

