import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set DagsHub Tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/lewko98/mlops-quickstart.mlflow"))

# Set experiment name
mlflow.set_experiment("mlops-quickstart")

with mlflow.start_run() as run:
    # Load dataset
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    
    # Train model
    n_estimators = 100
    max_depth = 3
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    # Log parameters and metrics
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", acc)
    
    # LOG THE MODEL ARTIFACT (This uploads the model to DagsHub!)
    mlflow.sklearn.log_model(clf, artifact_path="model")
    
    print(f"Run completed successfully! Run ID: {run.info.run_id}")
    print(f"Accuracy: {acc}")