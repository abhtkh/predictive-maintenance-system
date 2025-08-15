import sys
import os

# Add project root to sys.path for backend and ml imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from backend.inference import InferenceWrapper

app = FastAPI(title="Model Server")

inference_wrapper: InferenceWrapper = None

@app.on_event("startup")
async def startup_event():
    global inference_wrapper
    inference_wrapper = InferenceWrapper()

class InferenceRequest(BaseModel):
    sensor_values: List[float]

@app.post("/predict")
async def predict(request: InferenceRequest):
    result = inference_wrapper.predict(request.sensor_values)
    return result
