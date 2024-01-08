import pandas as pd
import src.services.data as data_

def preprocess_data(data):
    """Preprocess the data."""

    data = data.dropna()
    data = data.drop("Id", axis=1)
    data['Species'] = data['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
    data = data.rename(columns={"Species": "target"})
    data.to_csv("services\epf-flower-data-science\src\data\preprocessed_data.csv", index=False)

    return data.to_json(orient="records")