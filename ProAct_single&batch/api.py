from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from pydantic import BaseModel
import joblib
import os
import io
 
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model_path = os.path.join('models', 'model.joblib')
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    # Create a dummy model for testing
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit([[0,0,0,0]], [0])  # Dummy training
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, model_path)

class PredictionInput(BaseModel):
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Convert input data to format expected by model
        features = [[
            input_data.air_temperature,
            input_data.process_temperature,
            input_data.rotational_speed,
            input_data.torque
        ]]
        
        # Make prediction
        prediction = model.predict_proba(features)[0]
        
        return {
            "breakdown_probability": float(prediction[1]),
            "major_cause_name": "High Temperature" if prediction[1] > 0.5 else "Normal Operation"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded CSV file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Check if required columns exist
        required_columns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {', '.join(missing_columns)}"
            )
        
        # Make predictions
        features = df[required_columns].values
        
        # Handle any NaN values
        if np.isnan(features).any():
            features = np.nan_to_num(features, nan=0.0)
        
        predictions = model.predict_proba(features)
        
        # Create results DataFrame
        results_df = df.copy()
        results_df['breakdown_probability'] = predictions[:, 1]
        results_df['major_cause_name'] = ['High Temperature' if p > 0.5 else 'Normal Operation' for p in predictions[:, 1]]
        
        # Convert to JSON
        return results_df.to_dict(orient='records')
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
