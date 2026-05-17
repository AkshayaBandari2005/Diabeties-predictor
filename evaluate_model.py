import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import joblib

def calculate_metrics():
    print("Loading dataset and model...")
    df = pd.read_csv('diabetes_prediction_dataset.csv')
    
    try:
        model = joblib.load('diabetes_model.pkl')
    except FileNotFoundError:
        print("Error: 'diabetes_model.pkl' not found.")
        return

    print("Preprocessing and Predicting Probabilities...")
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Get probabilities
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Evaluate at multiple thresholds
    thresholds = [0.2, 0.3, 0.4, 0.5]
    
    print("\nModel Evaluation at Various Clinical Thresholds:")
    print("=" * 60)
    
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        report = classification_report(y_test, y_pred, output_dict=True)
        recall = report['1']['recall']
        precision = report['1']['precision']
        f1 = report['1']['f1-score']
        
        print(f"THRESHOLD: {t:.1f}")
        print(f"- Recall (Sensitivity): {recall*100:.1f}%")
        print(f"- Precision:            {precision*100:.1f}%")
        print(f"- F1-Score:             {f1:.3f}")
        print("-" * 30)

    # Confusion Matrix for the chosen threshold (0.3)
    t_final = 0.3
    y_pred_final = (y_probs >= t_final).astype(int)
    print(f"\nConfusion Matrix for Chosen Threshold ({t_final}):")
    print(confusion_matrix(y_test, y_pred_final))

if __name__ == '__main__':
    calculate_metrics()
