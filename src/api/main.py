import os
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import TransactionInput, RiskPrediction

app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk associated with transactions.",
    version="1.0.0"
)

# Global variable to store the loaded model
model = None

@app.on_event("startup")
def load_model():
    """
    Load the best model from the MLflow Model Registry on startup.
    """
    global model
    try:
        # Construct the model URI for the registered model
        # Using 'latest' or 'Production' alias if applicable
        model_name = "CreditRiskRandomForest"
        model_uri = f"models:/{model_name}/latest"
        
        print(f"Loading model from: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load model from MLflow: {e}")
        print("API will start but /predict will fail until a model is available.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Credit Risk Prediction API. Visit /docs for documentation."}

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "environment": os.getenv("ENV", "development")
    }

@app.post("/predict", response_model=RiskPrediction)
def predict(input_data: TransactionInput):
    """
    Predict risk for a given transaction.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Ensure MLflow server is accessible and model is registered.")
    
    try:
        # Convert Pydantic model to DataFrame for prediction
        input_dict = input_data.dict()
        df = pd.DataFrame([input_dict])
        
        # Perform prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        return RiskPrediction(
            is_high_risk=int(prediction),
            probability=float(probability),
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
