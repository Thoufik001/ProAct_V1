import requests
import json
import pandas as pd
import numpy as np
from tabulate import tabulate

def test_prediction(data):
    response = requests.post('http://localhost:8000/predict', json=data)
    return response.json()

# Generate synthetic test scenarios
def generate_test_scenarios():
    scenarios = []
    
    # 1. Normal Operations
    scenarios.append({
        "name": "Normal Operation - New Tool",
        "data": {
            "air_temperature": 298.0,  # 25°C
            "process_temperature": 308.0,  # 35°C
            "rotational_speed": 1500,
            "torque": 45.0,
            "tool_wear": 20,
            "type": 1
        }
    })
    
    scenarios.append({
        "name": "Normal Operation - Medium Wear",
        "data": {
            "air_temperature": 300.0,
            "process_temperature": 310.0,
            "rotational_speed": 1800,
            "torque": 50.0,
            "tool_wear": 150,
            "type": 1
        }
    })
    
    # 2. High Load Operations
    scenarios.append({
        "name": "High Speed - Low Wear",
        "data": {
            "air_temperature": 305.0,
            "process_temperature": 320.0,
            "rotational_speed": 2500,
            "torque": 60.0,
            "tool_wear": 50,
            "type": 2
        }
    })
    
    scenarios.append({
        "name": "High Torque - Medium Wear",
        "data": {
            "air_temperature": 308.0,
            "process_temperature": 325.0,
            "rotational_speed": 2000,
            "torque": 85.0,
            "tool_wear": 180,
            "type": 2
        }
    })
    
    # 3. Critical Conditions
    scenarios.append({
        "name": "Critical - High Temperature",
        "data": {
            "air_temperature": 318.0,
            "process_temperature": 345.0,
            "rotational_speed": 2200,
            "torque": 75.0,
            "tool_wear": 220,
            "type": 2
        }
    })
    
    scenarios.append({
        "name": "Critical - Extreme Wear",
        "data": {
            "air_temperature": 310.0,
            "process_temperature": 330.0,
            "rotational_speed": 2100,
            "torque": 80.0,
            "tool_wear": 290,
            "type": 2
        }
    })
    
    # 4. Edge Cases
    scenarios.append({
        "name": "Edge Case - All High",
        "data": {
            "air_temperature": 320.0,
            "process_temperature": 348.0,
            "rotational_speed": 2900,
            "torque": 95.0,
            "tool_wear": 295,
            "type": 2
        }
    })
    
    scenarios.append({
        "name": "Edge Case - Mixed",
        "data": {
            "air_temperature": 315.0,
            "process_temperature": 340.0,
            "rotational_speed": 2700,
            "torque": 90.0,
            "tool_wear": 250,
            "type": 1
        }
    })
    
    return scenarios

def main():
    print("Running Predictive Maintenance Tests\n")
    
    scenarios = generate_test_scenarios()
    results = []
    
    for scenario in scenarios:
        print(f"Testing: {scenario['name']}")
        response = test_prediction(scenario['data'])
        
        result = {
            "Scenario": scenario['name'],
            "Breakdown Probability": f"{response['breakdown_probability']:.2%}",
            "Prediction": "Breakdown" if response['breakdown_prediction'] == 1 else "Normal",
            "Confidence": response['confidence_level'],
            "Cause": response['major_cause_prediction']
        }
        results.append(result)
        
    # Print results in a table
    print("\nTest Results:")
    print(tabulate(results, headers="keys", tablefmt="grid"))
    
    # Calculate statistics
    breakdown_predictions = sum(1 for r in results if r['Prediction'] == 'Breakdown')
    print(f"\nStatistics:")
    print(f"Total Scenarios: {len(results)}")
    print(f"Predicted Breakdowns: {breakdown_predictions}")
    print(f"Predicted Normal Operations: {len(results) - breakdown_predictions}")

if __name__ == "__main__":
    main()
