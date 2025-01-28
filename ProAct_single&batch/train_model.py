import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, make_scorer, precision_recall_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
import numpy as np

def validate_data(dataset):
    required_columns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
                       'Torque [Nm]', 'Tool wear [min]', 'Type', 'Breakdown', 'Major_Cause']
    if not all(col in dataset.columns for col in required_columns):
        raise ValueError("Missing required columns in dataset")
    
    if dataset.isnull().any().any():
        raise ValueError("Dataset contains null values")
    
    if not (
        (200 <= dataset['Air temperature [K]'].min() <= dataset['Air temperature [K]'].max() <= 350) and
        (200 <= dataset['Process temperature [K]'].min() <= dataset['Process temperature [K]'].max() <= 350) and
        (0 <= dataset['Rotational speed [rpm]'].min() <= dataset['Rotational speed [rpm]'].max() <= 3000) and
        (0 <= dataset['Torque [Nm]'].min() <= dataset['Torque [Nm]'].max() <= 100) and
        (0 <= dataset['Tool wear [min]'].min() <= dataset['Tool wear [min]'].max() <= 300)
    ):
        raise ValueError("Data contains values outside expected ranges")

print("Loading and validating data...")
data_path = "data/predictive_maintenance.csv"
dataset = pd.read_csv(data_path)
validate_data(dataset)

# Preprocessing
print("\nPreprocessing data...")
le_type = LabelEncoder()
dataset['Type'] = le_type.fit_transform(dataset['Type'])
joblib.dump(le_type, "models/type_encoder.pkl")

le_cause = LabelEncoder()
dataset['Major_Cause'] = le_cause.fit_transform(dataset['Major_Cause'])
joblib.dump(le_cause, "models/cause_encoder.pkl")

# Features and targets
X = dataset[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
             'Torque [Nm]', 'Tool wear [min]', 'Type']]

# Add engineered features
X['temp_difference'] = dataset['Process temperature [K]'] - dataset['Air temperature [K]']
X['power_factor'] = dataset['Rotational speed [rpm]'] * dataset['Torque [Nm]'] / 1000
X['wear_rate'] = dataset['Tool wear [min]'] * dataset['Rotational speed [rpm]'] / 1000
X['stress_factor'] = X['power_factor'] * X['temp_difference'] / 100

y_breakdown = dataset['Breakdown']
y_cause = dataset['Major_Cause']

# Train/test split
X_train, X_test, y_breakdown_train, y_breakdown_test, y_cause_train, y_cause_test = train_test_split(
    X, y_breakdown, y_cause, test_size=0.2, random_state=42, stratify=y_breakdown
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save feature names and scaler
feature_names = list(X.columns)
joblib.dump(feature_names, "models/feature_names.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# Create pipelines
print("\nTraining breakdown model...")
breakdown_pipeline = Pipeline([
    ('smote', SMOTE(random_state=42, sampling_strategy=0.4)),  
    ('classifier', GradientBoostingClassifier(
        n_estimators=500,  
        learning_rate=0.05,  
        max_depth=8,  
        subsample=0.8,  
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    ))
])

cause_pipeline = Pipeline([
    ('smote', SMOTE(random_state=42, sampling_strategy='auto')),
    ('classifier', GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    ))
])

# Train and evaluate breakdown model
breakdown_pipeline.fit(X_train_scaled, y_breakdown_train)
y_pred_breakdown = breakdown_pipeline.predict(X_test_scaled)
y_pred_proba_breakdown = breakdown_pipeline.predict_proba(X_test_scaled)[:, 1]

print("\nBreakdown Model Performance:")
print(classification_report(y_breakdown_test, y_pred_breakdown))
print(f"ROC AUC Score: {roc_auc_score(y_breakdown_test, y_pred_proba_breakdown):.4f}")

# Find optimal threshold
precisions, recalls, thresholds = precision_recall_curve(y_breakdown_test, y_pred_proba_breakdown)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
optimal_idx = np.nanargmax(f1_scores)  
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal probability threshold: {optimal_threshold:.3f}")

# Train and evaluate cause model
print("\nTraining cause model...")
cause_pipeline.fit(X_train_scaled, y_cause_train)
y_pred_cause = cause_pipeline.predict(X_test_scaled)

print("\nCause Model Performance:")
print(classification_report(y_cause_test, y_pred_cause))

# Save models and metadata
print("\nSaving models and metadata...")
joblib.dump(breakdown_pipeline, "models/breakdown_model.pkl")
joblib.dump(cause_pipeline, "models/cause_model.pkl")
with open("models/threshold.txt", "w") as f:
    f.write(str(optimal_threshold))

print("Training completed successfully!")
