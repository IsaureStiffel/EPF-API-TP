from fastapi import HTTPException
from fastapi.responses import JSONResponse
from src.schemas.message import MessageResponse
import pandas as pd
import kaggle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import src.services.utils as utils
import src.services.cleaning as cleaning
import json
import joblib
from google.cloud import firestore


async def download_dataset():
    """Download the dataset from Kaggle."""

    kaggle.api.authenticate()
    
    dataset_name = "uciml/iris"
    save_dir = "EPF-API-TP/services/epf-flower-data-science/src/data/"
    kaggle.api.dataset_download_files(dataset_name, path=save_dir, unzip=True)

    return {"message": "Dataset downloaded and saved successfully."}

def load_iris_dataset():
    """Get the data from the dataset."""

    try:
        # Read the CSV file into a pandas DataFrame
        iris_df = pd.read_csv("services/epf-flower-data-science/src/data/Iris.csv")

        # Convert the DataFrame to JSON and return it
        json_data = iris_df.to_json(orient="records")

        # Return the JSON response
        return iris_df, json_data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Iris dataset file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load iris dataset: {str(e)}")
    
def preprocess_data():
    """Preprocess the data."""

    # Read the CSV file into a pandas DataFrame
    iris_df, json_data = load_iris_dataset()

    iris_preprocessed = cleaning.preprocess_data(iris_df)

    return iris_preprocessed
    
def split_iris_dataset():
    """Split the data into train and test sets."""

    data_preprocessed = pd.read_json(preprocess_data())
    data_train, data_test = train_test_split(data_preprocessed, test_size=0.2)
    return data_train.to_json(orient="records"), data_test.to_json(orient="records")
    
def train_classification_model():
    """Train the model."""

    # Read the splited dataset into a pandas DataFrame
    data_train = pd.read_json(split_iris_dataset()[0])
    X_train = data_train.drop("target", axis=1)
    y_train = data_train["target"]

    # Load model parameters from JSON file
    with open("services/epf-flower-data-science/src/config/model_parameters.json", "r") as file:
        model_parameters = json.load(file)

    model = RandomForestClassifier(model_parameters)
    model.fit(X_train, y_train)
    
    params_path = "services/epf-flower-data-science/src/config/"
    os.makedirs(params_path, exist_ok=True)
    params_file_path = os.path.join(params_path, "model_parameters.json")
    with open(params_file_path, "w") as f:
        json.dump(model.get_params(), f)

    model_path = "services/epf-flower-data-science/src/models/"
    os.makedirs(model_path, exist_ok=True)
    model_file_path = os.path.join(model_path, "model.joblib")
    joblib.dump(model, model_file_path)
    return MessageResponse(message="Model trained"), model

def predict_classification_model():
    """Predict the model."""
    model = train_classification_model()[1]
    data_test = pd.read_json(split_iris_dataset()[1])
    X_test = data_test.drop("target", axis=1)
    y_pred = model.predict(X_test)
    return pd.DataFrame(y_pred).to_json(orient="records")

def create_firestore():
    """Create a Firestore database."""
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/isaurestiffel/Doc_locaux/cours-EPF/2023-2024/DataSources/API_lab2/EPF-API-TP/services/epf-flower-data-science/src/config/credentials.json"
    db = firestore.Client()
    parameters_ref = db.collection('parameters').document('parameters')
    params = json.load(open("services/epf-flower-data-science/src/config/model_parameters.json"))
    parameters_ref.set(params)
    return MessageResponse(message="Firestore database created")

def get_data_firestore():
    """Get the data from Firestore."""
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/isaurestiffel/Doc_locaux/cours-EPF/2023-2024/DataSources/API_lab2/EPF-API-TP/services/epf-flower-data-science/src/config/credentials.json"
    db = firestore.Client()
    parameters_ref = db.collection('parameters').document('parameters')
    params = parameters_ref.get().to_dict()
    return params

def update_data_firestore():
    """Update the data from Firestore."""
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/isaurestiffel/Doc_locaux/cours-EPF/2023-2024/DataSources/API_lab2/EPF-API-TP/services/epf-flower-data-science/src/config/credentials.json"
    db = firestore.Client()
    parameters_ref = db.collection('parameters').document('parameters')
    params = parameters_ref.get().to_dict()
    params["n_estimators"] = 100
    parameters_ref.set(params)
    return MessageResponse(message="Firestore database updated")

