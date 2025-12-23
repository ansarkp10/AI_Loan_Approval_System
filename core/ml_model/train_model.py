import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

def create_sample_data():
    """Create sample training data"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.randint(20, 65, n_samples),
        'income': np.random.randint(20000, 200000, n_samples),
        'employment_type': np.random.choice(['salaried', 'self_employed', 'business', 'unemployed'], n_samples),
        'employment_years': np.random.randint(0, 40, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'marital_status': np.random.choice(['single', 'married', 'divorced'], n_samples),
        'dependents': np.random.randint(0, 5, n_samples),
        'loan_amount': np.random.randint(5000, 500000, n_samples),
        'loan_type': np.random.choice(['personal', 'home', 'car', 'education', 'business'], n_samples),
        'tenure_months': np.random.choice([12, 24, 36, 48, 60, 120, 240], n_samples),
        'debt_to_income_ratio': np.random.uniform(0.1, 0.8, n_samples),
        'loan_to_value_ratio': np.random.uniform(0.1, 0.9, n_samples),
        'existing_loans': np.random.randint(0, 5, n_samples),
        'existing_loan_amount': np.random.randint(0, 200000, n_samples),
    }
    
    # Create target variable based on logical rules
    df = pd.DataFrame(data)
    
    # Simulate approval logic
    conditions = (
        (df['credit_score'] > 650) &
        (df['debt_to_income_ratio'] < 0.5) &
        (df['income'] > 30000) &
        (df['employment_years'] > 2)
    )
    
    df['loan_approved'] = np.where(conditions, 1, 0)
    
    # Add some noise
    noise = np.random.choice([0, 1], n_samples, p=[0.1, 0.9])
    df['loan_approved'] = df['loan_approved'] ^ noise
    
    return df

def train_and_save_model():
    """Train ML model and save it"""
    print("Creating sample data...")
    df = create_sample_data()
    
    # Prepare features and target
    X = df.drop('loan_approved', axis=1)
    y = df['loan_approved']
    
    # Encode categorical variables
    categorical_cols = ['employment_type', 'marital_status', 'loan_type']
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['age', 'income', 'employment_years', 'credit_score', 
                     'dependents', 'loan_amount', 'tenure_months',
                     'debt_to_income_ratio', 'loan_to_value_ratio',
                     'existing_loans', 'existing_loan_amount']
    
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Train Random Forest model
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    rf_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and preprocessing objects
    joblib.dump(rf_model, 'core/ml_model/loan_model.pkl')
    joblib.dump(scaler, 'core/ml_model/scaler.pkl')
    joblib.dump(label_encoders, 'core/ml_model/label_encoders.pkl')
    
    print("\nModel and preprocessing objects saved successfully!")
    
    return rf_model, scaler, label_encoders, accuracy

def predict_loan_approval(data):
    """Predict loan approval for new data"""
    try:
        # Load model and preprocessing objects
        model = joblib.load('core/ml_model/loan_model.pkl')
        scaler = joblib.load('core/ml_model/scaler.pkl')
        label_encoders = joblib.load('core/ml_model/label_encoders.pkl')
        
        # Prepare input data
        input_df = pd.DataFrame([data])
        
        # Encode categorical variables
        for col, le in label_encoders.items():
            if col in input_df.columns:
                # Handle unseen labels
                if input_df[col].iloc[0] in le.classes_:
                    input_df[col] = le.transform([input_df[col].iloc[0]])[0]
                else:
                    input_df[col] = -1  # Unknown label
        
        # Scale numerical features
        numerical_cols = ['age', 'income', 'employment_years', 'credit_score', 
                         'dependents', 'loan_amount', 'tenure_months',
                         'debt_to_income_ratio', 'loan_to_value_ratio',
                         'existing_loans', 'existing_loan_amount']
        
        # Ensure all numerical columns exist
        for col in numerical_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        # Calculate risk score (0-100)
        risk_score = (1 - probability) * 100
        
        return {
            'approved': bool(prediction),
            'probability': float(probability),
            'risk_score': float(risk_score),
            'status': 'Approved' if prediction else 'Rejected'
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

if __name__ == "__main__":
    train_and_save_model()