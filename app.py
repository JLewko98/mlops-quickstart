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