import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import joblib

def main():
    print("Loading dataset...")
    df = pd.read_csv('diabetes_prediction_dataset.csv')
    
    # Define features
    numeric_features = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    categorical_features = ['gender', 'smoking_history']
    
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']
    
    # --- SPLIT FIRST TO AVOID DATA LEAKAGE ---
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Reconstruct training df for upsampling
    train_df = pd.concat([X_train_raw, y_train_raw], axis=1)
    
    # --- HANDLING IMBALANCE VIA UPSAMPLING (ONLY ON TRAIN SET) ---
    df_majority = train_df[train_df.diabetes == 0]
    df_minority = train_df[train_df.diabetes == 1]
    
    print(f"Original Training distribution: No Diabetes={len(df_majority)}, Diabetes={len(df_minority)}")
    
    # Upsample minority to match majority
    df_minority_upsampled = resample(df_minority, 
                                     replace=True,     
                                     n_samples=len(df_majority),    
                                     random_state=42)
    
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    
    print(f"Balanced Training distribution: No Diabetes={len(df_balanced[df_balanced.diabetes==0])}, Diabetes={len(df_balanced[df_balanced.diabetes==1])}")
    
    X_train = df_balanced.drop('diabetes', axis=1)
    y_train = df_balanced['diabetes']
    
    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # EXTRA TREES CLASSIFIER is often better for imbalanced recall optimization
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', ExtraTreesClassifier(random_state=42, n_jobs=-1, class_weight={0: 1, 1: 5}))
    ])
    
    # Hyperparameter search space
    param_dist = {
        'classifier__n_estimators': [100, 200, 300, 400],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2', None]
    }
    
    print("Starting Optimized Randomized Search on balanced learning data...")
    search = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_dist, 
        n_iter=5, # Reduced for speed as requested
        cv=3,      # Reduced for speed as requested
        scoring='f1_macro', 
        verbose=1, 
        n_jobs=-1,
        random_state=42
    )


    
    search.fit(X_train, y_train)
    
    model = search.best_estimator_
    
    print("\nEvaluating Optimized Extra Trees on SEPARATE Test Set...")
    y_pred = model.predict(X_test)
    print("\nClassification Report (Threshold 0.5):")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nSaving maximized recall pipeline...")
    joblib.dump(model, 'diabetes_model.pkl')
    print("Optimization finished successfully. Saved 'diabetes_model.pkl'.")

if __name__ == '__main__':
    main()




