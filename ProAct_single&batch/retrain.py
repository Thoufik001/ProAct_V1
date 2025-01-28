import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, make_scorer
from imblearn.over_sampling import SMOTE
import joblib
import numpy as np

def automated_retraining(new_data_path, existing_data_path):
    # Load and merge data
    new_data = pd.read_csv("data/new_data.csv")
    existing_data = pd.read_csv("data/existing_data.csv")
    full_data = pd.concat([existing_data, new_data], ignore_index=True)

    # Preprocessing
    le_type = LabelEncoder()
    full_data['Type'] = le_type.fit_transform(full_data['Type'])

    le_cause = LabelEncoder()
    full_data['Major_Cause'] = le_cause.fit_transform(full_data['Major_Cause'])

    X = full_data[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
                   'Torque [Nm]', 'Tool wear [min]', 'Type']]
    y_breakdown = full_data['Breakdown']
    y_cause = full_data['Major_Cause']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_smote_breakdown, y_smote_breakdown = smote.fit_resample(X_scaled, y_breakdown)
    X_smote_cause, y_smote_cause = smote.fit_resample(X_scaled, y_cause)

    # Train Breakdown Model
    breakdown_model = GradientBoostingClassifier(random_state=42)
    breakdown_model.fit(X_smote_breakdown, y_smote_breakdown)

    # Train Cause Model
    cause_model = GradientBoostingClassifier(random_state=42)
    cause_model.fit(X_smote_cause, y_smote_cause)

    # Cross-Validation
    def cross_validate_model(model, X, y, scoring, cv_splits=5):
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
        return scores

    # Validate Breakdown Model
    cv_scores_breakdown = cross_validate_model(
        breakdown_model, X_smote_breakdown, y_smote_breakdown,
        scoring=make_scorer(roc_auc_score, needs_proba=True)
    )
    print(f"Breakdown Model CV AUC Scores: {cv_scores_breakdown}")
    print(f"Breakdown Model Mean AUC: {np.mean(cv_scores_breakdown):.4f}")

    # Validate Cause Model
    cv_scores_cause = cross_validate_model(
        cause_model, X_smote_cause, y_smote_cause,
        scoring=make_scorer(roc_auc_score, needs_proba=True, multi_class='ovo', average='weighted')
    )
    print(f"Cause Model CV AUC Scores: {cv_scores_cause}")
    print(f"Cause Model Mean AUC: {np.mean(cv_scores_cause):.4f}")

    
    joblib.dump(breakdown_model, "models/breakdown_model.pkl")
    joblib.dump(cause_model, "models/cause_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    
    full_data.to_csv(existing_data_path, index=False)
    print("Retraining completed and models updated.")
