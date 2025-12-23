# django_integration.py
import joblib
import pandas as pd
import numpy as np

class LoanPredictor:
    def __init__(self):
        # Load trained model and preprocessing objects
        self.model = joblib.load('models/loan_approval_model.pkl')
        self.scaler = joblib.load('models/scaler.pkl')
        self.label_encoders = joblib.load('models/label_encoders.pkl')
        self.feature_columns = joblib.load('models/feature_columns.pkl')
    
    def preprocess_input(self, input_data):
        """Preprocess input data for prediction"""
        # Create DataFrame from input
        df = pd.DataFrame([input_data])
        
        # Handle missing values
        df = df.fillna({
            'Gender': 'Male',
            'Married': 'No',
            'Dependents': '0',
            'Self_Employed': 'No',
            'LoanAmount': df['LoanAmount'].median() if 'LoanAmount' in df.columns else 0,
            'Loan_Amount_Term': 360,
            'Credit_History': 1.0
        })
        
        # Feature engineering (same as training)
        if 'ApplicantIncome' in df.columns and 'CoapplicantIncome' in df.columns:
            df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        
        if 'LoanAmount' in df.columns and 'TotalIncome' in df.columns:
            df['Loan_to_Income_Ratio'] = df['LoanAmount'] / df['TotalIncome']
        
        if 'LoanAmount' in df.columns and 'Loan_Amount_Term' in df.columns:
            df['EMI'] = df['LoanAmount'] * 0.085 * (1 + 0.085)**df['Loan_Amount_Term'] / \
                       ((1 + 0.085)**df['Loan_Amount_Term'] - 1)
        
        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                # Handle unseen labels
                if df[col].iloc[0] in encoder.classes_:
                    df[col] = encoder.transform([df[col].iloc[0]])[0]
                else:
                    df[col] = -1  # Unknown label
        
        # Ensure all required columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns to match training
        df = df[self.feature_columns]
        
        return df
    
    def predict(self, input_data):
        """Make prediction for a single loan application"""
        # Preprocess input
        processed_data = self.preprocess_input(input_data)
        
        # Scale features
        scaled_data = self.scaler.transform(processed_data)
        
        # Make prediction
        prediction = self.model.predict(scaled_data)[0]
        probability = self.model.predict_proba(scaled_data)[0][1]
        
        # Calculate risk score
        risk_score = (1 - probability) * 100
        
        return {
            'approved': bool(prediction),
            'probability': float(probability),
            'risk_score': float(risk_score),
            'status': 'Approved' if prediction else 'Rejected'
        }

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = LoanPredictor()
    
    # Test with sample data
    sample_application = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '2',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 50000,
        'CoapplicantIncome': 20000,
        'LoanAmount': 200000,
        'Loan_Amount_Term': 360,
        'Credit_History': 1.0,
        'Property_Area': 'Urban'
    }
    
    result = predictor.predict(sample_application)
    print("Prediction Result:")
    print(f"Status: {result['status']}")
    print(f"Approval Probability: {result['probability']:.2%}")
    print(f"Risk Score: {result['risk_score']:.2f}")