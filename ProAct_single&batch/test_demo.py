import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Load the models and encoders
breakdown_model = joblib.load('models/breakdown_model.pkl')
cause_model = joblib.load('models/cause_model.pkl')
scaler = joblib.load('models/scaler.pkl')
le_type = joblib.load('models/type_encoder.pkl')
le_cause = joblib.load('models/cause_encoder.pkl')

# Create demo data
def generate_demo_data(n_samples=10):
    np.random.seed(42)
    
    demo_data = {
        'Air temperature [K]': np.random.uniform(295, 305, n_samples),
        'Process temperature [K]': np.random.uniform(305, 315, n_samples),
        'Rotational speed [rpm]': np.random.uniform(1300, 1500, n_samples),
        'Torque [Nm]': np.random.uniform(30, 70, n_samples),
        'Tool wear [min]': np.random.uniform(0, 250, n_samples),
        'Type': np.random.choice(['L', 'M', 'H'], n_samples)
    }
    
    df = pd.DataFrame(demo_data)
    
    # Add engineered features
    df['temp_difference'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['power_factor'] = df['Rotational speed [rpm]'] * df['Torque [Nm]'] / 1000
    df['wear_rate'] = df['Tool wear [min]'] / 100
    df['stress_factor'] = df['Torque [Nm]'] * df['Tool wear [min]'] / 1000
    
    return df

def make_predictions(data):
    # Prepare features
    X = data.copy()
    X['Type'] = le_type.transform(X['Type'])
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    breakdown_prob = breakdown_model.predict_proba(X_scaled)[:, 1]
    breakdown_pred = breakdown_model.predict(X_scaled)
    
    cause_pred = cause_model.predict(X_scaled)
    cause_names = le_cause.inverse_transform(cause_pred)
    
    return breakdown_prob, breakdown_pred, cause_names

def visualize_results(data, breakdown_prob, breakdown_pred, cause_names):
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Breakdown Probability
    sns.barplot(x=range(len(breakdown_prob)), y=breakdown_prob, ax=ax1)
    ax1.set_title('Breakdown Probability for Each Sample')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Probability')
    ax1.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
    ax1.legend()
    
    # Plot 2: Feature Importance
    features = ['Air Temp', 'Process Temp', 'Speed', 'Torque', 'Tool Wear', 
                'Temp Diff', 'Power Factor', 'Wear Rate', 'Stress Factor']
    feature_values = data[['Air temperature [K]', 'Process temperature [K]', 
                          'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
                          'temp_difference', 'power_factor', 'wear_rate', 'stress_factor']].values
    
    # Normalize feature values for visualization
    feature_values = (feature_values - feature_values.min(axis=0)) / (feature_values.max(axis=0) - feature_values.min(axis=0))
    
    sns.heatmap(feature_values.T, xticklabels=range(len(breakdown_prob)), 
                yticklabels=features, ax=ax2, cmap='YlOrRd')
    ax2.set_title('Normalized Feature Values')
    
    plt.tight_layout()
    plt.savefig('demo_results.png')
    
    # Print predictions
    print("\nPrediction Results:")
    print("-" * 50)
    for i in range(len(breakdown_pred)):
        print(f"\nSample {i}:")
        print(f"Breakdown Probability: {breakdown_prob[i]:.2%}")
        print(f"Predicted Breakdown: {'Yes' if breakdown_pred[i] else 'No'}")
        print(f"Predicted Cause: {cause_names[i]}")

def main():
    print("Generating demo data...")
    demo_data = generate_demo_data()
    
    print("Making predictions...")
    breakdown_prob, breakdown_pred, cause_names = make_predictions(demo_data)
    
    print("Visualizing results...")
    visualize_results(demo_data, breakdown_prob, breakdown_pred, cause_names)
    
    print("\nResults have been saved to 'demo_results.png'")

if __name__ == "__main__":
    main()
