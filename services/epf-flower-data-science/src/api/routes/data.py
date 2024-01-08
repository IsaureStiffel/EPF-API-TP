from fastapi import APIRouter
from fastapi.responses import JSONResponse
import src.services.data as service

router = APIRouter()

@router.get("/download-dataset")
async def download_dataset():
    return service.download_dataset()


@router.get("/load_iris")
def load_iris_dataset():
    iris_df, json_data = service.load_iris_dataset()
    return JSONResponse(content=json_data, media_type="application/json")

@router.get("/preprocess_data")
def preprocess_data():
    return service.preprocess_data()
    
@router.get("/split_dataset")
def split_iris_dataset():
    return service.split_iris_dataset()

@router.get("/train_model")
def train_classification_model():
    return service.train_classification_model()

@router.get("/predict_model")
def predict_model_router():
    return service.predict_classification_model()

@router.get("/create_firestore")
def create_firestore_router():
    return service.create_firestore()

@router.get("/get_data_firestore")
def get_data_firestore_router():
    return service.get_data_firestore()

@router.put("/update_data_firestore/")
def update_data_firestore_router():
    return service.update_data_firestore()