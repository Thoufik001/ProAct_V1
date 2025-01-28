from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import io
from retrain import automated_retraining
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='batch_prediction_debug.log')

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

breakdown_model = joblib.load("models/breakdown_model.pkl")
cause_model = joblib.load("models/cause_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

cause_mapping = {
    0: "No Failure",
    1: "Power Failure",
    2: "Tool Wear Failure",
    3: "Overstrain Failure",
    4: "Random Failures",
    5: "Other"
}

class MaintenanceInput(BaseModel):
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float
    type: int

def process_single_prediction(input_dict, scaler, breakdown_model, cause_model):
    """
    Process a single prediction with more flexible input handling
    """
    try:
        # Remove any additional columns not needed for prediction
        prediction_features = [
            'air_temperature', 'process_temperature', 
            'rotational_speed', 'torque', 'tool_wear', 'type'
        ]
        
        # Ensure only required features are used
        input_data = {k: input_dict[k] for k in prediction_features}
        
        # Convert input to the correct format for prediction
        input_array = np.array([
            input_data['air_temperature'],
            input_data['process_temperature'],
            input_data['rotational_speed'],
            input_data['torque'],
            input_data['tool_wear'],
            input_data['type']
        ]).reshape(1, -1)
        
        # Scale the input
        input_scaled = scaler.transform(input_array)
        
        # Predict breakdown probability
        breakdown_prob = breakdown_model.predict_proba(input_scaled)[0][1]
        breakdown_prediction = 1 if breakdown_prob > 0.4 else 0
        
        # Predict major cause
        cause_prob = cause_model.predict_proba(input_scaled)
        major_cause = np.argmax(cause_prob)
        
        return {
            "breakdown_probability": float(breakdown_prob),
            "breakdown_prediction": int(breakdown_prediction),
            "major_cause_prediction": int(major_cause),
            "major_cause_name": cause_mapping.get(major_cause, "Unknown")
        }
    except Exception as e:
        logging.error(f"Error processing prediction: {str(e)}")
        raise ValueError(f"Error processing prediction: {str(e)}")

import traceback

def preprocess_batch_data(df):
    """
    Comprehensive data preprocessing for batch prediction
    
    Handles:
    - Column name variations
    - Type conversion
    - Feature engineering
    """
    # Comprehensive column mapping
    column_mapping = {
        'air_temperature': ['Air temperature [K]', 'Air Temperature', 'AirTemp'],
        'process_temperature': ['Process temperature [K]', 'Process Temperature', 'ProcessTemp'],
        'rotational_speed': ['Rotational speed [rpm]', 'Rotational Speed', 'RotSpeed'],
        'torque': ['Torque [Nm]', 'Torque', 'TorqueNm'],
        'tool_wear': ['Tool wear [min]', 'Tool Wear', 'ToolWear'],
        'type': ['Type', 'Machine Type', 'MachineType']
    }
    
    # Find matching columns
    matched_columns = {}
    for target, possible_names in column_mapping.items():
        found_column = next((col for col in df.columns if col in possible_names), None)
        if found_column:
            matched_columns[target] = found_column
    
    # Validate required columns
    required_columns = ['air_temperature', 'process_temperature', 
                        'rotational_speed', 'torque', 'tool_wear', 'type']
    
    missing_cols = [col for col in required_columns if col not in matched_columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}\n"
            f"Available columns: {list(df.columns)}\n"
            f"Matched columns: {matched_columns}"
        )
    
    # Create a copy of the dataframe
    df_processed = df.copy()
    
    # Rename columns
    df_processed.rename(columns={
        matched_columns['air_temperature']: 'air_temperature',
        matched_columns['process_temperature']: 'process_temperature',
        matched_columns['rotational_speed']: 'rotational_speed',
        matched_columns['torque']: 'torque',
        matched_columns['tool_wear']: 'tool_wear',
        matched_columns['type']: 'type'
    }, inplace=True)
    
    # Type mapping
    type_mapping = {'L': 0, 'M': 1, 'H': 2}
    
    # Convert type to numeric
    df_processed['type'] = df_processed['type'].map(type_mapping)
    
    # Validate type mapping
    if df_processed['type'].isnull().any():
        invalid_types = df_processed[df_processed['type'].isnull()]['type'].unique()
        raise ValueError(f"Invalid machine types found: {invalid_types}. Expected 'L', 'M', or 'H'.")
    
    # Feature engineering
    df_processed['temp_difference'] = df_processed['process_temperature'] - df_processed['air_temperature']
    df_processed['power_factor'] = df_processed['rotational_speed'] * df_processed['torque'] / 1000
    df_processed['wear_rate'] = df_processed['tool_wear'] * df_processed['rotational_speed'] / 1000
    df_processed['stress_factor'] = df_processed['power_factor'] * df_processed['temp_difference'] / 100
    
    # Select and order columns
    columns_order = [
        'air_temperature', 'process_temperature', 'rotational_speed', 
        'torque', 'tool_wear', 'type', 
        'temp_difference', 'power_factor', 'wear_rate', 'stress_factor'
    ]
    
    return df_processed[columns_order]

def batch_prediction_pipeline(df):
    """
    Complete batch prediction pipeline
    """
    try:
        # Preprocess the data
        df_processed = preprocess_batch_data(df)
        
        # Separate features for scaling and prediction
        X = df_processed.values
        
        # Scale the input data
        X_scaled = scaler.transform(X)
        
        # Predict breakdown probabilities
        breakdown_probs = breakdown_model.predict_proba(X_scaled)[:, 1]
        breakdown_predictions = (breakdown_probs > 0.4).astype(int)
        
        # Predict causes
        cause_probs = cause_model.predict_proba(X_scaled)
        cause_predictions = np.argmax(cause_probs, axis=1)
        
        # Prepare results
        results = []
        for i in range(len(df_processed)):
            result = {
                "breakdown_probability": float(breakdown_probs[i]),
                "breakdown_prediction": int(breakdown_predictions[i]),
                "major_cause_prediction": int(cause_predictions[i]),
                "major_cause_name": cause_mapping.get(cause_predictions[i], "Unknown"),
                **df_processed.iloc[i].to_dict()
            }
            results.append(result)
        
        # Calculate summary statistics
        summary = {
            "total_samples": len(results),
            "breakdown_predictions": sum(1 for r in results if r["breakdown_prediction"] == 1),
            "cause_distribution": pd.Series([r["major_cause_name"] for r in results]).value_counts().to_dict(),
            "avg_breakdown_probability": np.mean([r["breakdown_probability"] for r in results]),
            "high_risk_samples": sum(1 for r in results if r["breakdown_probability"] > 0.7)
        }
        
        return {
            "results": results,
            "summary": summary
        }
    
    except Exception as e:
        logging.error(f"Batch prediction error: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(input_data: MaintenanceInput):
    try:
        # Validate input ranges
        if not (200 <= input_data.air_temperature <= 350 or  # Typical range for industrial processes in Kelvin
                200 <= input_data.process_temperature <= 350 or
                0 <= input_data.rotational_speed <= 3000 or  # Typical RPM range
                0 <= input_data.torque <= 100 or            # Typical torque range
                0 <= input_data.tool_wear <= 300):          # Typical tool wear in minutes
            raise ValueError("Input values are outside expected ranges")

        # Prepare input data with base features
        base_data = np.array([[input_data.air_temperature, input_data.process_temperature,
                              input_data.rotational_speed, input_data.torque,
                              input_data.tool_wear, input_data.type]])
        
        # Calculate engineered features
        temp_difference = input_data.process_temperature - input_data.air_temperature
        power_factor = input_data.rotational_speed * input_data.torque / 1000
        wear_rate = input_data.tool_wear * input_data.rotational_speed / 1000
        stress_factor = power_factor * temp_difference / 100
        
        # Combine all features
        data = np.column_stack([
            base_data,
            np.array([[temp_difference, power_factor, wear_rate, stress_factor]])
        ])
        
        # Scale the input data
        try:
            data_scaled = scaler.transform(data)
        except Exception as e:
            raise ValueError(f"Scaling error: Input data format doesn't match training data: {str(e)}")

        # Make predictions with probability threshold
        breakdown_prob = breakdown_model.predict_proba(data_scaled)[0, 1]
        breakdown_pred = 1 if breakdown_prob >= 0.4 else 0  # Less conservative threshold
        
        # Get cause prediction and probability
        cause_probs = cause_model.predict_proba(data_scaled)[0]
        cause_pred = cause_model.predict(data_scaled)[0]
        
        # If there's a breakdown predicted, ensure we don't predict "No Failure"
        if breakdown_pred == 1 and cause_pred == 0:
            # Get the next most likely cause
            cause_probs[0] = 0  # Zero out "No Failure" probability
            cause_pred = np.argmax(cause_probs)

        # Return detailed response
        response = {
            "breakdown_probability": float(breakdown_prob),
            "breakdown_prediction": int(breakdown_pred),
            "major_cause_prediction": cause_mapping.get(cause_pred, "Unknown"),
            "major_cause_code": int(cause_pred),
            "cause_probabilities": {cause_mapping[i]: float(p) for i, p in enumerate(cause_probs)},
            "confidence_level": "High" if abs(breakdown_prob - 0.5) > 0.3 else "Medium" if abs(breakdown_prob - 0.5) > 0.15 else "Low",
            "parameters": {
                "temp_difference": float(temp_difference),
                "power_factor": float(power_factor),
                "wear_rate": float(wear_rate),
                "stress_factor": float(stress_factor)
            }
        }
        
        return response

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    """
    Batch prediction endpoint
    """
    try:
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Run batch prediction pipeline
        prediction_results = batch_prediction_pipeline(df)
        
        return prediction_results
    
    except Exception as e:
        logging.error(f"Batch prediction endpoint error: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
def retrain(new_data_path: str, existing_data_path: str):
    try:
        automated_retraining(new_data_path, existing_data_path)
        return {"message": "Retraining completed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Saved the current state of the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
