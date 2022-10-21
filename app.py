from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel
from preprocessing.cleaning_data import preprocess
from predict.prediction import prediction
import json
import numpy as np


# with open('input_data.json') as json_file1:
#     input_data = json.load(json_file1)

class House(BaseModel):

    area: int
    property_type: Literal["APARTMENT","Data","OTHERS"]
    rooms_number: int
    zip_code: int
    land_area: int | None = None
    garden: bool | None = None
    garden_area: int | None = None
    equipped_kitchen: bool | None = None
    full_address: str | None = None
    swimming_pool: bool | None = None
    furnished: bool | None = None
    open_fire: bool | None = None
    terrace: bool | None = None
    terrace_area: int | None = None
    facades_number: int | None = None
    building_state: Literal["NEW", "GOOD", "TO RENOVATE", "JUST RENOVATED", "TO REBUILD"] | None = None


class House_data(BaseModel):

    data : House

class Predict(BaseModel):

    prediction: float | None = None
    status_code: int | None = None


default_dict = '''
  {"data": {
    "area": int,
    "property-type": ["APARTMENT","HOUSE","OTHERS"],
    "rooms-number": int,
    "zip-code": int,
    "land-area": Optional[int],
    "garden": Optional[bool],
    "garden-area": Optional[int],
    "equipped-kitchen": Optional[bool],
    "full-address": Optional[str],
    "swimming-pool": Optional[bool],
    "furnished": Optional[bool],
    "open-fire": Optional[bool],
    "terrace": Optional[bool],
    "terrace-area": Optional[int],
    "facades-number": Optional[int]
    # "building-state": [Optional[str] == ["NEW","GOOD","TO RENOVATE","JUST RENOVATED","TO REBUILD"]]
  }}
  '''

app = FastAPI()

@app.get("/")
def root():
    return {"Alive!"}


@app.post("/predict", response_model=Predict)
def post_predict(house_data : House_data):
    features = np.array(preprocess(house_data))
    features = features.reshape(1,-1)
    predict = prediction(features)
    price_prediction = Predict(prediction=predict, status_code=201)
    return price_prediction


@app.get("/predict")
def schema():
    return default_dict

# price = post_predict(input_data)
# print(price)