import pandas as pd
import joblib

def test_pipeline():
    print("Loading model...")
    model = joblib.load('diabetes_model.pkl')
    
    # Test data matching CSV order
    # gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level
    test_input = pd.DataFrame([{
        'gender': 'Female',
        'age': 50.0,
        'hypertension': 0,
        'heart_disease': 0,
        'smoking_history': 'never',
        'bmi': 27.3,
        'HbA1c_level': 6.5,
        'blood_glucose_level': 180
    }])
    
    print("\nInput Data:")
    print(test_input)
    
    print("\nPredicting...")
    pred = model.predict(test_input)[0]
    prob = model.predict_proba(test_input)[0][1]
    
    print(f"Prediction: {pred} (Diabetes: {'YES' if pred==1 else 'NO'})")
    print(f"Probability: {prob*100:.2f}%")

if __name__ == '__main__':
    test_pipeline()
