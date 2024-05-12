from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
import pandas as pd
import uvicorn
from tensorflow.keras.models import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

NON_CUSTOM_MODEL = load("models/non_custom_prediction_pipeline.joblib")
CUSTOM_MODEL = load("models/custom_prediction_pipeline.joblib")
PRICE_FORECASTING_MODEL = load_model("models/price_forecasting_model.h5")


@app.post("/api/v1")
async def on_message(request: Request):
    try:
        body = await request.json()
        series = pd.DataFrame({"culture": [body["culture"]], "humidity": [body["humidity"]], "region": [body["region"]], "superficie": [body["superficie"]], "temp": [body["temp"]], "rainfall": [body["rainfall"]], "wind": [body["wind"]]})
        prediction = NON_CUSTOM_MODEL.predict(series)
    except Exception as e:
            return {
            "status": "KO",
            "error": repr(e)
        }
    else:
        return {
            "rendement": prediction[0],
            "status": "OK"
        }


@app.post("/api/v2")
async def on_message(request: Request):
    try:
        body = await request.json()
        series = pd.DataFrame({"culture": [body["culture"]], "humidity": [body["humidity"]], "region": [body["region"]], "superficie": [body["superficie"]], "temp": [body["temp"]], "rainfall": [body["rainfall"]], "wind": [body["wind"]], "N": [body["N"]], "P": [body["P"]], "K": [body["K"]], "ph": [body["ph"]]})
        prediction = CUSTOM_MODEL.predict(series)  
    except Exception as e:
            return {
            "status": "KO",
            "error": repr(e)
        }
    else:
        return {
            "rendement": prediction[0],
            "status": "OK"
        }  


@app.post("/api/price_forecast")
async def on_message(request: Request):
    try:
        body = await request.json()
        series = pd.DataFrame({"regions": [body["regions"]], "date": [body["date"]], "produits": [body["produits"]]})
        prediction = PRICE_FORECASTING_MODEL.predict(series)  
    except Exception as e:
            return {
            "status": "KO",
            "error": repr(e)
        }
    else:
        return {
            "prix/KG": prediction[0],
            "status": "OK"
        } 

if __name__=="__main__":
    uvicorn.run("app:app",host="0.0.0.0",port=5005)