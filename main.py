from fastapi import FastAPI
from fastapi import status
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from joblib import load

class Features(BaseModel):
    season:int
    mnth:int
    holiday:int
    weekday:int
    workingday:int
    weathersit:int
    temp:float
    atemp:float
    hum:float
    windspeed:float

    class Config:
        schema_extra = {
            "example":{
                "season": 3.,
                "mnth": 7.,
                "holiday": 0.,
                "weekday": 6.,
                "workingday": 0,
                "weathersit":1,
                "temp": 0.686667,
                "atemp": 0.638263,
                "hum": 0.585,
                "windspeed": 0.208342
            }
        }
class Label(BaseModel):
    rentals: float

class Message(BaseModel):
    message: float


app = FastAPI()

@app.post(
    "/Rentas/",
    response_model=Label,
    status_code=status.HTTP_202_ACCEPTED,
    summary="PREDICCION DE RENTAS",
    description="Prediccion de rentas",
    tags=["Rentals"]
)
async def get_rentals(features:Features):
    try:
        model = load('model.joblib')
        data = [
            features.season,
            features.mnth,
            features.holiday,
            features.weekday,
            features.workingday,
            features.weathersit,
            features.temp,
            features.atemp,
            features.hum,
            features.windspeed
        ]
        predictions = model.predict([data])
        response = {"rentals":predictions[0]}
        return response
    except Exception as e:
        response = JSONResponse(
                    status_code=400,
                    content={"message":f"{e.args}"},
                )   
        return response 