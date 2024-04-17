import uvicorn
from fastapi import FastAPI
from patientInfo import p_info
import numpy as np
import pickle
import pandas as pd

#Creating the app object
app = FastAPI()
pickle_in = open("model_final.pkl","rb")
classifier = pickle.load(pickle_in)

# Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': "This is Robin's Claims prediction assignment"}

# @app.get('/{name}')
# def get_name(name: str):
#     return {'message': f'Hello, {name}'}

#Code to get Disease Score
from patientInfo import diseaseScore
@app.post('/disease_score')
async def diseaseSc(payload : diseaseScore):
    data = payload.__dict__
    print(data)
    # print("Hello")

    # Extract values from the dictionary
    features = [data[field] for field in diseaseScore.model_fields.keys()]

    count = 10
    score = 0
    for feature in features:
        if(feature == 1):
            score += count
        count -= 1

    print("Your disease score is : ", score)

    return {
        'Desease Score' : score
    }


#Code to get Physician Score
from patientInfo import PhysicianScore
@app.post('/physician_score')
async def PhysicianSc(payload : PhysicianScore):
    data = payload.__dict__
    print(data)
    # print("Hello")

    # Extract values from the dictionary
    features = [data[field] for field in PhysicianScore.model_fields.keys()]

    count = 15
    score = 0
    for feature in features:
        if(feature == 1):
            score += count
        count -= 5

    print("Your Physician score is : ", score)

    return {
        'Physician Score' : score
    }


#Code to get Procedure Score
from patientInfo import ProcedureScore
@app.post('/procedure_score')
async def PhysicianSc(payload : ProcedureScore):
    data = payload.__dict__
    print(data)

    # Extract values from the dictionary
    features = [data[field] for field in ProcedureScore.model_fields.keys()]

    count = 6
    score = 0
    for feature in features:
        if(feature == 1):
            score += count
        count -= 1

    print("Your Procedure score is : ", score)

    return {
        'Procedure Score' : score
    }

#Code to get Procedure Score
from patientInfo import DiagnosisScore
@app.post('/diagnosis_score')
async def PhysicianSc(payload : DiagnosisScore):
    data = payload.__dict__
    print(data)

    # Extract values from the dictionary
    features = [data[field] for field in DiagnosisScore.model_fields.keys()]

    count = 10
    score = 0
    for feature in features:
        if(feature == 1):
            score += count
        count -= 1

    print("Your Diagnosis score is : ", score)

    return {
        'Diagnosis Score' : score
    }



@app.post('/predict')
async def predict(payload : p_info):
    data = payload.__dict__
    print(data)

    # Extract values from the dictionary
    features = [data[field] for field in p_info.model_fields.keys()]

    # Make prediction
    prediction = classifier.predict([features])[0]

    print("Predicted insurance amount is : ", prediction)

    return {
        'Prediction' : prediction
    }




# Running the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1', port=5000)

#Command to run : python -m uvicorn app:app --reload