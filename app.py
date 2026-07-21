import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np

# Pydantic schema for inference requests
class PredictRequest(BaseModel):
    features: list[float]

# Global model state holder
model_container = {}

def _fetch_mlflow_model():
    """Synchronous helper function to download model artifacts without locking the event loop."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/lewko98/mlops-quickstart.mlflow")
    run_id = os.getenv("MLFLOW_RUN_ID", "c7f26cd19fe3412083d12dcccc9b52db")
    
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"runs:/{run_id}/random_forest_model"
    
    print(f"--> Connecting to DagsHub MLflow ({tracking_uri})...")
    print(f"--> Fetching model run artifact ({model_uri})...")
    
    loaded_model = mlflow.sklearn.load_model(model_uri)
    print("--> Model successfully loaded into memory!")
    return loaded_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Non-blocking startup: execute blocking MLflow network calls in a background thread
    try:
        model_container["model"] = await asyncio.to_thread(_fetch_mlflow_model)
    except Exception as e:
        print(f"ERROR: Failed to load model from MLflow: {e}")
        model_container["model"] = None
    yield
    # Cleanup on shutdown
    model_container.clear()

app = FastAPI(title="MLOps Model Inference API", lifespan=lifespan)

@app.get("/health")
def health_check():
    if model_container.get("model") is None:
        return {"status": "initializing", "message": "Model loading in progress or failed"}
    return {"status": "healthy"}

@app.post("/predict")
def predict(request: PredictRequest):
    model = model_container.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready or failed to load")
    
    data = np.array(request.features).reshape(1, -1)
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}