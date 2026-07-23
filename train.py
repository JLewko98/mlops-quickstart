import os
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    # 1. Initialize DagsHub repository integration
    # Reads DAGSHUB_USER_TOKEN or MLFLOW_TRACKING_PASSWORD from environment variables
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "JLewko98")
    repo_name = os.getenv("DAGSHUB_REPO_NAME", "mlops-quickstart")
    
    print(f"--> Initializing DagsHub tracking for {repo_owner}/{repo_name}...")
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

    # 2. Enable automatic logging for scikit-learn
    mlflow.sklearn.autolog(log_models=True)

    # 3. Load and split dataset
    print("--> Loading Iris dataset...")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    # 4. Start MLflow Run
    print("--> Starting MLflow training run...")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"\n==================================================")
        print(f" SUCCESS! MLflow Active Run ID: {run_id}")
        print(f"==================================================\n")

        # Train model
        n_estimators = 100
        max_depth = 5
        
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate model
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions, average="weighted")
        rec = recall_score(y_test, predictions, average="weighted")
        f1 = f1_score(y_test, predictions, average="weighted")

        # Explicitly log key evaluation metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        print(f"--> Model Accuracy:  {acc:.4f}")
        print(f"--> Model Precision: {prec:.4f}")
        print(f"--> Model Recall:    {rec:.4f}")
        print(f"--> Model F1-Score:  {f1:.4f}")
        
        print("\n--> Training complete. Artifacts successfully logged to DagsHub!")

if __name__ == "__main__":
    main()
