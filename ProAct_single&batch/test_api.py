import requests
import json

# Test cases
test_cases = [
    {
        "air_temperature": 298.0,
        "process_temperature": 308.0,
        "rotational_speed": 1420,
        "torque": 45.0,
        "tool_wear": 120,
        "type": 0
    },
    {
        "air_temperature": 310.0,
        "process_temperature": 330.0,
        "rotational_speed": 2200,
        "torque": 85.0,
        "tool_wear": 250,
        "type": 1
    },
    {
        "air_temperature": 295.0,
        "process_temperature": 305.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 50,
        "type": 2
    }
]

def test_prediction_endpoint():
    url = "http://localhost:8000/predict"
    headers = {"Content-Type": "application/json"}
    
    print("Testing Predictive Maintenance API\n")
    print("-" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print("Input Parameters:")
        for key, value in test_case.items():
            print(f"  {key}: {value}")
        
        try:
            response = requests.post(url, json=test_case, headers=headers)
            if response.status_code == 200:
                result = response.json()
                print("\nPrediction Results:")
                print(f"  Breakdown Probability: {result['breakdown_probability']:.2%}")
                print(f"  Breakdown Prediction: {'Yes' if result['breakdown_prediction'] == 1 else 'No'}")
                print(f"  Major Cause: {result['major_cause_prediction']}")
                print(f"  Confidence Level: {result['confidence_level']}")
            else:
                print(f"\nError: {response.status_code}")
                print(response.json())
        except Exception as e:
            print(f"\nError making request: {str(e)}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_prediction_endpoint()
