from typing import List, Dict, Union
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableParallel
from langchain_core.prompts import format_document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents.base import DEFAULT_DOCUMENT_PROMPT
from glob import glob
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from meteostat import Stations
from neuralprophet import NeuralProphet, set_log_level, load
import pandas as pd
import warnings
from joblib import load
from fastapi.requests import Request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanAbsoluteError
import numpy as np

warnings.filterwarnings('ignore')
set_log_level("ERROR")

NON_CUSTOM_MODEL = load("app/models/non_custom_prediction_pipeline.joblib")
CUSTOM_MODEL = load("app/models/custom_prediction_pipeline.joblib")
ENCODER = load("app/models/encoder.pkl")
PRICE_MODEL = load_model("app/models/price_prediction.h5", custom_objects={'mae': MeanAbsoluteError()})
DISEASE_MODEL = load_model("/app/models/cnn_model_save.h5")

# function 
def format_docs(inputs: dict) -> str:
  inputs["context"] = "\n\n".join(
      f"Document {i}:\n {format_document(doc, DEFAULT_DOCUMENT_PROMPT)}" for i, doc in enumerate(inputs["context"])
  )
  return inputs

def format_chat_history(data):
  formatize_chat_history = ""
  if "chat_history" in data.keys():
    for message in data["chat_history"]:
      message_type = str(type(message)).split("'")[1].split(".")[-1]
      message_content = message.content.replace("\n", "")
      formatize_chat_history += f"\t{message_type}: {message_content}\n"
    data["chat_history"] = formatize_chat_history
  return data

def format_response(response: str) -> dict:
  return {"input": response}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Définition du modèle de données d'entrée pour le endpoint
class WeatherData(BaseModel):
    latitude: float
    longitude: float
    data: List[Dict[str, Union[str, float]]]

# ingestion
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

pages = []
for document in glob(pathname="app/documents/*"):
  loader = PyPDFLoader(document)
  pages.extend(loader.load_and_split(text_splitter=splitter))

# llm
llm = ChatOpenAI(
  openai_api_base="https://api.groq.com/openai/v1",
  model = "llama3-70b-8192",
  temperature=0.7,
  api_key="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"
)

# indexing
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
vectorstore = Chroma.from_documents(documents=pages, embedding=embeddings)

# retriever
retriever = vectorstore.as_retriever()

# QA Prompt
qa_system_prompt = """
Je suis un assistant agricole intelligent qui fournit des recommandations sur l'irrigation, la fertilisation, les traitements 
phytosanitaires et les meilleures pratiques agricoles. Sur la base du contexte, des informations de l'utilisateur et la question 
de l'utilisateur, fournit des recommandations à l'utilisateur de manière précise sur comment il pourrait agir pour améliorer son
rendement.

contexte : {context}
"""
#informations: {info}

qa_prompt = ChatPromptTemplate.from_messages(
  [
    ("system", "Je suis Mbaykat🌾🦾🤖🧠 une solution de recommandations agricoles basée sur l'IA Générative."),
    ("system", qa_system_prompt),
    ("human", "{input}"),
  ]
)

# Contextualization
contextualize_query_system_prompt = """
Je suis un assistant IA très util pour contextualiser les fils de discusssion. Compte tenu de l’historique du chat et de la 
dernière question de l’utilisateur, formule une question autonome qui peut être compris sans l’historique du chat.
Donne juste la question directe et brute qui sera la proche possible de la dernière question de l'utilisateur.

NE REPONDEZ PAS A LA QUESTION, reformulez-la au besoin ou retournez-la tel quel sinon.

Historique_Chat: 
{chat_history}

Dernière_Question: {input}
"""

contextualize_query_prompt = ChatPromptTemplate.from_template(contextualize_query_system_prompt)

contextualization_chain = RunnableBranch(
  (
    lambda x: not x.get("chat_history", False) ,
    lambda x: x["input"],
  ),
  format_chat_history | contextualize_query_prompt | llm | StrOutputParser(),
)

retrieval_chain = {"context": retriever , "input": RunnablePassthrough()}

chain = contextualization_chain | retrieval_chain | qa_prompt | llm | StrOutputParser()

store = {}

conversational_rag_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

CLASS_NAME_DISEASE = [
  'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy'
]

def predict(model, img):
  img_array = tf.keras.preprocessing.image.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0)
  predictions = model.predict(img_array)
  predicted_class = CLASS_NAME_DISEASE[np.argmax(predictions[0])]
  return predicted_class

app = FastAPI()

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

@app.get("/api/forecast_temperature")
def predict_temperature(weather_data: WeatherData):
    try:
        # Extraction des données d'entrée
        latitude = weather_data.latitude
        longitude = weather_data.longitude
        data_list = weather_data.data

        # Convert data_list to DataFrame
        data = pd.DataFrame(data_list)

        # Convert 'ds' column to datetime type
        data['ds'] = pd.to_datetime(data['ds'])
        
        # Get nearby weather stations
        stations = Stations()

        station = stations.nearby(latitude, longitude)
        station = stations.fetch().sort_values("distance").iloc[0]
        
        if station.country != "SN":
            raise HTTPException(
                status_code=400, 
                detail="Cette zone n'est pas prise en charge par le modèle de prédiction."
            )
        
        model_path = f"app/models/{station.wmo}.np"

        # Initialisation du modèle NeuralProphet
        model = NeuralProphet()

        # Load le modèle
        model = load(model_path)

        # Préparation des données pour la prédiction
        future = model.make_future_dataframe(data, periods=3)

        # Prédiction des températures pour les 3 prochaines heures
        forecast = model.predict(future)

        # Filtrage des prédictions pour les 3 prochaines heures
        forecast_next_3_hours = forecast.tail(3)
        
        # Récupération des timestamps des prédictions
        timestamps = pd.to_datetime(forecast_next_3_hours['ds'].values)
        
        # Récupération des températures prédites
        predicted_temperatures = forecast_next_3_hours['yhat1'].values.tolist()

        # Création d'une liste de dictionnaires pour représenter les prédictions
        predictions = [{"timestamp": ts.strftime('%Y-%m-%d %H:00:00').replace('T', ' '), "temperature": int(temp)} for ts, temp in zip(timestamps, predicted_temperatures)]

        # Renvoi des prédictions de températures avec les timestamps
        return {"predictions": predictions}
    
    except HTTPException as e:
        raise e
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/yield-forecast/v1")
async def predict_yield_v1(request: Request):
    try:
      body = await request.json()
      series = pd.DataFrame({"culture": [body["culture"]], "humidity": [body["humidity"]], "region": [body["region"]], "superficie": [body["superficie"]], "temp": [body["temp"]], "rainfall": [body["rainfall"]], "wind": [body["wind"]]})
      prediction = NON_CUSTOM_MODEL.predict(series)
    except Exception as e:
      return HTTPException(status_code=500, detail=repr(e))
    else:
      return {
          "rendement": prediction[0],
          "status": "OK"
      }


@app.post("/api/yield-forecast/v2")
async def predict_yield_v2(request: Request):
    try:
      body = await request.json()
      series = pd.DataFrame({"culture": [body["culture"]], "humidity": [body["humidity"]], "region": [body["region"]], "superficie": [body["superficie"]], "temp": [body["temp"]], "rainfall": [body["rainfall"]], "wind": [body["wind"]], "N": [body["N"]], "P": [body["P"]], "K": [body["K"]], "ph": [body["ph"]]})
      prediction = CUSTOM_MODEL.predict(series)
    except Exception as e:
      return HTTPException(status_code=500, detail=repr(e))
    else:
      return {
          "rendement": prediction[0],
          "status": "OK"
      }
    
@app.post("/api/price-forecast")
async def predict_price(request: Request):
    try:
      body = await request.json()
      series = pd.DataFrame({
          "produits": [body["produit"]],
          "regions": [body["region"]],
          "mois": [body["mois"]],
          "annee": [body["annee"]],
          "prix/KG": [body["prix"]]
      })
      series[["produits", "regions"]] = ENCODER.transform(series[["produits","regions"]])
      X = np.array(series.values, dtype=np.float32)
      X = np.expand_dims(X, axis=1)
      prediction = PRICE_MODEL.predict(X)[0][0]
    except Exception as e:
      return HTTPException(status_code=500, detail=repr(e))
    else:
      return {
          "prix": float(prediction.astype(np.float64)),
          "status": "OK"
      }
    
@app.post('/api/disease')
async def get_disease(request: Request):
    form = await request.form()
    file_content = await form.get("file").read()
    try:
        img = tf.image.decode_image(file_content)
        predicted_class = predict(DISEASE_MODEL, img.numpy())
    except Exception as e:
        return HTTPException(status_code=500, detail={"error", repr(e)})
    else:
      if predicted_class == "Tomato___healthy":
        result = "Healthy"
      else:
        result = "Diseased"
      return {
        "status": "OK",
        "result": result
      }

# Edit this to add the chain you want to add
add_routes(app, conversational_rag_chain, path="/mbaykat")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)