import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

# Load model from MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
model_name = "Advanced_Medical_Cost_Prediction"
model_stage = "Production"

model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_stage}")

## Note: If model is not regestered successfully, you need to manually register it with code below

# from mlflow.tracking import MlflowClient

# # Initialize MLflow client
# client = MlflowClient()

# # Set the stage for the registered model version
# client.transition_model_version_stage(
#     name="Advanced_Medical_Cost_Prediction",
#     version=1,  # make sure this is the version you want to promote
#     stage="Production"  # Options: "Staging", "Production", "Archived"
# )

# print("Model version successfully transitioned to 'Production'.")


# Define input schema
class InsuranceInput(BaseModel):
    age: int
    sex: str          # "male" or "female"
    bmi: float
    children: int
    smoker: str       # "yes" or "no"
    region: str       # "southeast", etc.

# Init FastAPI
app = FastAPI(title="Medical Insurance Cost Prediction")

# Encode categorical fields manually
def preprocess_input(data: InsuranceInput):
    df = pd.DataFrame([data.dict()])
    df['sex'] = df['sex'].map({'male': 1, 'female': 0})
    df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
    df['region'] = df['region'].map({
        'southwest': 0,
        'southeast': 1,
        'northwest': 2,
        'northeast': 3
    })
    return df

@app.post("/predict")
def predict(data: InsuranceInput):
    try:
        df = preprocess_input(data)
        prediction = model.predict(df)
        return {"predicted_medical_cost": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the server
if __name__ == "__main__":
    uvicorn.run("Medical_Insurance_FastAPI_Deployment:app", host="0.0.0.0", port=8000, reload=True)


# Steps:
# 1. Run mlflow ui in one terminal:
# mlflow ui --host 127.0.0.1 --port 5000
# 2. Register and promote your model (done via MLflow UI or script with MlflowClient).
# 3. Install required packages:
# pip install fastapi uvicorn pandas mlflow scikit-learn

#  To launch the API server: (Make sure your FastAPI app is running)
# uvicorn Medical_Insurance_FastAPI_Deployment:app --host 0.0.0.0 --port 8000 --reload

# Open your browser and go to: http://127.0.0.1:8000/docs

# Test JSON Again in Swagger:
# Click on POST /predict
# Click the Try it out button, then paste this JSON in the request body:

# {
#   "age": 45,
#   "sex": "male",
#   "bmi": 25.3,
#   "children": 2,
#   "smoker": "no",
#   "region": "southeast"
# }

# Click Execute – you’ll get a prediction in the response. 
# e.g. http://127.0.0.1:8000/predict   
# result: {"detail":"Method Not Allowed"}
