# core/ml_integration.py
"""
Machine Learning integration for Loan Approval System
"""

import joblib
import pandas as pd
import numpy as np
import os
from pathlib import Path

class LoanPredictor:
    """Loan approval predictor using trained ML model"""
    
    def __init__(self):
        """Initialize predictor"""
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = []
        self.model_loaded = False
        
        try:
            # Try to load the trained model
            self.load_model()
            self.model_loaded = True
            print("✅ ML Model loaded successfully!")
        except Exception as e:
            print(f"⚠️  ML Model not loaded: {e}")
            self.model_loaded = False
    
    def load_model(self):
        """Load trained model and preprocessing objects"""
        # Define paths relative to project root
        base_dir = Path(__file__).resolve().parent.parent
        
        model_files = {
            'model': base_dir / 'models' / 'loan_approval_model.pkl',
            'scaler': base_dir / 'models' / 'scaler.pkl',
            'encoders': base_dir / 'models' / 'label_encoders.pkl',
            'features': base_dir / 'models' / 'feature_columns.pkl'
        }
        
        # Check if model files exist
        for name, path in model_files.items():
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load all model files
        self.model = joblib.load(model_files['model'])
        self.scaler = joblib.load(model_files['scaler'])
        self.label_encoders = joblib.load(model_files['encoders'])
        self.feature_columns = joblib.load(model_files['features'])
    
    def preprocess_application(self, application_data):
        """Preprocess loan application for ML prediction"""
        # Create DataFrame
        df = pd.DataFrame([application_data])
        
        # Fill missing values
        defaults = {
            'Gender': 'Male',
            'Married': 'No',
            'Dependents': '0',
            'Education': 'Graduate',
            'Self_Employed': 'No',
            'LoanAmount': 0,
            'Loan_Amount_Term': 360,
            'Credit_History': 1.0,
            'Property_Area': 'Urban',
            'ApplicantIncome': 0,
            'CoapplicantIncome': 0
        }
        
        for key, default in defaults.items():
            if key not in df.columns or pd.isna(df[key].iloc[0]):
                df[key] = default
        
        # Feature engineering
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        
        if df['TotalIncome'].iloc[0] > 0:
            df['Loan_to_Income_Ratio'] = df['LoanAmount'] / df['TotalIncome']
        else:
            df['Loan_to_Income_Ratio'] = 0
        
        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                value = str(df[col].iloc[0])
                if value in encoder.classes_:
                    df[col] = encoder.transform([value])[0]
                else:
                    df[col] = 0
        
        # Ensure all required columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns
        df = df[self.feature_columns]
        
        return df
    
    def predict(self, application_data):
        """Predict loan approval"""
        if not self.model_loaded:
            return self.get_fallback_prediction(application_data)
        
        try:
            # Preprocess
            processed_data = self.preprocess_application(application_data)
            
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
                'status': 'Approved' if prediction else 'Rejected',
                'confidence': 'High' if probability > 0.8 else 'Medium' if probability > 0.6 else 'Low'
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self.get_fallback_prediction(application_data)
    
    def get_fallback_prediction(self, application_data):
        """Fallback prediction when model is not loaded"""
        # Simple rule-based fallback
        income = application_data.get('ApplicantIncome', 0) + application_data.get('CoapplicantIncome', 0)
        loan_amount = application_data.get('LoanAmount', 0)
        
        # Simple rule: approve if loan amount < 5x income
        if income > 0 and loan_amount > 0:
            approved = loan_amount / income < 5
            probability = 0.7 if approved else 0.3
            risk_score = 30 if approved else 70
        else:
            approved = False
            probability = 0.3
            risk_score = 70
        
        return {
            'approved': approved,
            'probability': probability,
            'risk_score': risk_score,
            'status': 'Approved' if approved else 'Rejected (Fallback)',
            'confidence': 'Low',
            'note': 'Using rule-based fallback'
        }

# Create a global instance
predictor_instance = None

def get_predictor():
    """Get or create predictor instance"""
    global predictor_instance
    if predictor_instance is None:
        predictor_instance = LoanPredictor()
    return predictor_instance