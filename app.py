import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn

# Global model container
model = None
model_loading_error = None

def _fetch_mlflow_model():
    """Synchronous helper function to download model artifacts without locking the event loop."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/lewko98/mlops-quickstart.mlflow")
    run_id = os.getenv("MLFLOW_RUN_ID", "8533cbff79744f758abc03fbf37e9549")
    
    mlflow.set_tracking_uri(tracking_uri)
    
    # Try 'model' first (used by train.py), fallback to others
    artifact_paths = ["model", "random_forest_model", "iris_model"]
    
    for path in artifact_paths:
        model_uri = f"runs:/{run_id}/{path}"
        print(f"--> Trying to fetch model artifact from: {model_uri}...")
        try:
            loaded_model = mlflow.sklearn.load_model(model_uri)
            print(f"--> Success! Loaded model from artifact path '{path}'.")
            return loaded_model
        except Exception as e:
            print(f"--> Path '{path}' failed: {e}")
            
    raise Exception(f"Could not find a valid MLflow model artifact in run {run_id} using paths: {artifact_paths}")

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Asynchronous lifespan handler to load model in the background on startup."""
    global model, model_loading_error
    print("--> Application starting up: Loading MLflow model in background thread...")
    try:
        model = await asyncio.to_thread(_fetch_mlflow_model)
        print("--> Model successfully loaded into memory and ready for predictions!")
    except Exception as e:
        model_loading_error = str(e)
        print(f"--> ERROR: Failed to load model from MLflow: {e}")
    yield
    print("--> Application shutting down...")

# Instantiate FastAPI application
app = FastAPI(title="MLOps Iris Prediction API", lifespan=lifespan)

class FeaturesInput(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    if model is not None:
        return {"status": "healthy"}
    elif model_loading_error:
        return {"status": "error", "message": f"Model failed to load: {model_loading_error}"}
    else:
        return {"status": "initializing", "message": "Model loading in progress"}

@app.post("/predict")
def predict(data: FeaturesInput):
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail=f"Model not loaded. Status: {model_loading_error or 'initializing'}"
        )
    try:
        prediction = model.predict([data.features])
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")