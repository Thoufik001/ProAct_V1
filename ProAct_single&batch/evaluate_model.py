import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report, roc_auc_score
)
import joblib
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_prepare_data():
    # Load the dataset
    data = pd.read_csv("data/predictive_maintenance.csv")
    
    # Prepare features
    X = data[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
              'Torque [Nm]', 'Tool wear [min]', 'Type']]
    
    # Encode machine type
    le_type = joblib.load("models/type_encoder.pkl")
    X['Type'] = le_type.transform(X['Type'])
    
    # Add engineered features
    X['temp_difference'] = data['Process temperature [K]'] - data['Air temperature [K]']
    X['power_factor'] = data['Rotational speed [rpm]'] * data['Torque [Nm]'] / 1000
    X['wear_rate'] = data['Tool wear [min]'] * data['Rotational speed [rpm]'] / 1000
    X['stress_factor'] = X['power_factor'] * X['temp_difference'] / 100
    
    # Prepare targets
    y_breakdown = data['Breakdown']
    y_cause = data['Major_Cause']
    
    # Encode cause labels
    le_cause = joblib.load("models/cause_encoder.pkl")
    y_cause = le_cause.transform(y_cause)
    
    return X, y_breakdown, y_cause

def evaluate_breakdown_model(X, y_true, model, scaler):
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Get predictions
    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    false_alarm_rate = fp / (fp + tn)
    miss_rate = fn / (fn + tp)
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall (Sensitivity)': recall,
        'F1 Score': f1,
        'AUC-ROC': auc_roc,
        'Specificity': specificity,
        'False Alarm Rate': false_alarm_rate,
        'Miss Rate': miss_rate,
        'Confusion Matrix': cm
    }

def evaluate_cause_model(X, y_true, model, scaler):
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Get predictions
    y_pred = model.predict(X_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro')
    weighted_precision = precision_score(y_true, y_pred, average='weighted')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    weighted_recall = recall_score(y_true, y_pred, average='weighted')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Get detailed classification report
    class_report = classification_report(y_true, y_pred)
    
    return {
        'Accuracy': accuracy,
        'Macro Precision': macro_precision,
        'Weighted Precision': weighted_precision,
        'Macro Recall': macro_recall,
        'Weighted Recall': weighted_recall,
        'Macro F1': macro_f1,
        'Weighted F1': weighted_f1,
        'Confusion Matrix': cm,
        'Classification Report': class_report
    }

def plot_confusion_matrices(breakdown_cm, cause_cm, cause_labels):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot breakdown confusion matrix
    sns.heatmap(breakdown_cm, annot=True, fmt='d', ax=ax1)
    ax1.set_title('Breakdown Prediction Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_xticklabels(['No Breakdown', 'Breakdown'])
    ax1.set_yticklabels(['No Breakdown', 'Breakdown'])
    
    # Plot cause confusion matrix
    sns.heatmap(cause_cm, annot=True, fmt='d', ax=ax2)
    ax2.set_title('Cause Prediction Confusion Matrix')
    ax2.set_xlabel('Predicted Cause')
    ax2.set_ylabel('Actual Cause')
    ax2.set_xticklabels(cause_labels, rotation=45)
    ax2.set_yticklabels(cause_labels, rotation=45)
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.close()

def main():
    print("Loading models and data...")
    # Load models and scaler
    breakdown_model = joblib.load("models/breakdown_model.pkl")
    cause_model = joblib.load("models/cause_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    
    # Load and prepare data
    X, y_breakdown, y_cause = load_and_prepare_data()
    
    print("\nEvaluating Breakdown Prediction Model:")
    breakdown_metrics = evaluate_breakdown_model(X, y_breakdown, breakdown_model, scaler)
    
    print("\nBreakdown Prediction Metrics:")
    for metric, value in breakdown_metrics.items():
        if metric != 'Confusion Matrix':
            print(f"{metric}: {value:.4f}")
    
    print("\nBreakdown Confusion Matrix:")
    print(breakdown_metrics['Confusion Matrix'])
    
    print("\nEvaluating Major Cause Prediction Model:")
    cause_metrics = evaluate_cause_model(X, y_cause, cause_model, scaler)
    
    print("\nMajor Cause Prediction Metrics:")
    for metric, value in cause_metrics.items():
        if metric not in ['Confusion Matrix', 'Classification Report']:
            print(f"{metric}: {value:.4f}")
    
    print("\nDetailed Classification Report for Cause Prediction:")
    print(cause_metrics['Classification Report'])
    
    # Plot confusion matrices
    cause_labels = ['No Failure', 'Power Failure', 'Tool Wear Failure', 
                   'Overstrain Failure', 'Random Failures', 'Other']
    plot_confusion_matrices(
        breakdown_metrics['Confusion Matrix'],
        cause_metrics['Confusion Matrix'],
        cause_labels
    )
    
    print("\nConfusion matrices have been saved to 'confusion_matrices.png'")

if __name__ == "__main__":
    main()
