import os
import pandas as pd
import numpy as np

# Create necessary directories
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Create sample dataset with realistic correlations
n_samples = 1000
np.random.seed(42)

# Base features
data = {
    'Air temperature [K]': np.random.uniform(280, 320, n_samples),  # Normal operating range
    'Process temperature [K]': None,  # Will be calculated
    'Rotational speed [rpm]': np.random.uniform(1000, 2800, n_samples),
    'Torque [Nm]': np.random.uniform(20, 80, n_samples),
    'Tool wear [min]': np.random.uniform(0, 300, n_samples),
    'Type': np.random.choice(['L', 'M', 'H'], n_samples, p=[0.3, 0.5, 0.2])  # More medium-type machines
}

# Create correlations
df = pd.DataFrame(data)

# Process temperature correlates with air temperature and rotational speed
df['Process temperature [K]'] = df['Air temperature [K]'] + \
                               0.1 * df['Rotational speed [rpm]']/2800 * \
                               (df['Torque [Nm]']/80) * \
                               np.random.uniform(5, 15, n_samples)

# Calculate breakdown probability based on conditions
breakdown_prob = (
    0.1 +  # base probability
    0.3 * (df['Tool wear [min]'] > 200).astype(float) +  # high tool wear
    0.2 * ((df['Process temperature [K]'] - df['Air temperature [K]']) > 30).astype(float) +  # high temp difference
    0.2 * (df['Rotational speed [rpm]'] > 2500).astype(float) +  # high speed
    0.2 * (df['Torque [Nm]'] > 70).astype(float) +  # high torque
    0.1 * (df['Type'] == 'H').astype(float)  # heavy-duty machines
)

# Ensure probability is between 0 and 1
breakdown_prob = np.clip(breakdown_prob, 0, 1)

# Generate breakdowns based on probability
df['Breakdown'] = np.random.binomial(1, breakdown_prob)

# Generate causes based on conditions
def determine_cause(row):
    if row['Breakdown'] == 0:
        return 'No Failure'
    
    probs = {
        'Power Failure': 0.1,
        'Tool Wear Failure': 0.2,
        'Overstrain Failure': 0.2,
        'Random Failures': 0.1,
        'Other': 0.1
    }
    
    # Adjust probabilities based on conditions
    if row['Tool wear [min]'] > 200:
        probs['Tool Wear Failure'] += 0.3
    if row['Torque [Nm]'] > 70:
        probs['Overstrain Failure'] += 0.3
    if row['Process temperature [K]'] - row['Air temperature [K]'] > 30:
        probs['Power Failure'] += 0.2
        
    # Normalize probabilities
    total = sum(probs.values())
    probs = {k: v/total for k, v in probs.items()}
    
    return np.random.choice(list(probs.keys()), p=list(probs.values()))

df['Major_Cause'] = df.apply(determine_cause, axis=1)

# Save the dataset
df.to_csv("data/predictive_maintenance.csv", index=False)
print("Setup completed successfully!")
