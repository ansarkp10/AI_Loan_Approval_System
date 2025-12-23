# train_model_kaggle.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os

class LoanApprovalModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
    def load_data(self, train_path='train.csv', test_path='test.csv'):
        """Load Kaggle loan prediction datasets"""
        print("Loading Kaggle Loan Prediction datasets...")
        
        # Load training data
        self.train_df = pd.read_csv(train_path)
        print(f"Training data shape: {self.train_df.shape}")
        
        # Load test data (for final evaluation)
        self.test_df = pd.read_csv(test_path)
        print(f"Test data shape: {self.test_df.shape}")
        
        # Display dataset info
        print("\nTraining Data Info:")
        print(self.train_df.info())
        
        print("\nTraining Data Columns:")
        print(self.train_df.columns.tolist())
        
        print("\nTraining Data Sample:")
        print(self.train_df.head())
        
        return self.train_df, self.test_df
    
    def explore_data(self):
        """Explore and analyze the dataset"""
        print("\n" + "="*60)
        print("DATASET EXPLORATION")
        print("="*60)
        
        # Check target variable
        if 'Loan_Status' in self.train_df.columns:
            print(f"\n1. Loan Status Distribution:")
            print(self.train_df['Loan_Status'].value_counts())
            print(f"Approval Rate: {(self.train_df['Loan_Status'] == 'Y').mean():.2%}")
        
        # Check missing values
        print("\n2. Missing Values in Training Data:")
        missing_train = self.train_df.isnull().sum()
        missing_train = missing_train[missing_train > 0]
        print(missing_train)
        
        print("\nMissing Values in Test Data:")
        missing_test = self.test_df.isnull().sum()
        missing_test = missing_test[missing_test > 0]
        print(missing_test)
        
        # Display basic statistics
        print("\n3. Numerical Features Statistics:")
        numerical_cols = self.train_df.select_dtypes(include=[np.number]).columns
        print(self.train_df[numerical_cols].describe())
        
        return self.train_df, self.test_df
    
    def preprocess_data(self):
        """Preprocess the Kaggle loan dataset"""
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)
        
        # Create a copy for preprocessing
        train_df = self.train_df.copy()
        test_df = self.test_df.copy()
        
        # Handle missing values
        print("\n1. Handling missing values...")
        
        # Fill missing values for numerical columns
        numerical_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
        for col in numerical_cols:
            if col in train_df.columns:
                median_val = train_df[col].median()
                train_df[col] = train_df[col].fillna(median_val)
                if col in test_df.columns:
                    test_df[col] = test_df[col].fillna(median_val)
        
        # Fill missing values for categorical columns
        categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed']
        for col in categorical_cols:
            if col in train_df.columns:
                mode_val = train_df[col].mode()[0]
                train_df[col] = train_df[col].fillna(mode_val)
                if col in test_df.columns:
                    test_df[col] = test_df[col].fillna(mode_val)
        
        # Feature Engineering
        print("\n2. Feature engineering...")
        
        # Create total income
        if 'ApplicantIncome' in train_df.columns and 'CoapplicantIncome' in train_df.columns:
            train_df['TotalIncome'] = train_df['ApplicantIncome'] + train_df['CoapplicantIncome']
            test_df['TotalIncome'] = test_df['ApplicantIncome'] + test_df['CoapplicantIncome']
        
        # Create loan to income ratio
        if 'LoanAmount' in train_df.columns and 'TotalIncome' in train_df.columns:
            train_df['Loan_to_Income_Ratio'] = train_df['LoanAmount'] / train_df['TotalIncome']
            test_df['Loan_to_Income_Ratio'] = test_df['LoanAmount'] / test_df['TotalIncome']
        
        # Create EMI (simplified calculation)
        if 'LoanAmount' in train_df.columns and 'Loan_Amount_Term' in train_df.columns:
            train_df['EMI'] = train_df['LoanAmount'] * 0.085 * (1 + 0.085)**train_df['Loan_Amount_Term'] / \
                            ((1 + 0.085)**train_df['Loan_Amount_Term'] - 1)
            test_df['EMI'] = test_df['LoanAmount'] * 0.085 * (1 + 0.085)**test_df['Loan_Amount_Term'] / \
                           ((1 + 0.085)**test_df['Loan_Amount_Term'] - 1)
        
        # Encode categorical variables
        print("\n3. Encoding categorical variables...")
        
        # Define categorical columns to encode
        cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 
                   'Self_Employed', 'Property_Area']
        
        for col in cat_cols:
            if col in train_df.columns:
                le = LabelEncoder()
                # Fit on training data
                le.fit(train_df[col].astype(str))
                self.label_encoders[col] = le
                
                # Transform both train and test
                train_df[col] = le.transform(train_df[col].astype(str))
                if col in test_df.columns:
                    test_df[col] = le.transform(test_df[col].astype(str))
        
        # Prepare features and target
        if 'Loan_Status' in train_df.columns:
            # Convert target to binary
            train_df['Loan_Status'] = train_df['Loan_Status'].map({'Y': 1, 'N': 0})
            
            # Drop unnecessary columns
            drop_cols = ['Loan_ID']
            features_df = train_df.drop(drop_cols + ['Loan_Status'], axis=1, errors='ignore')
            target_df = train_df['Loan_Status']
        else:
            features_df = train_df.drop(['Loan_ID'], axis=1, errors='ignore')
            target_df = None
        
        # Prepare test features
        test_features = test_df.drop(['Loan_ID'], axis=1, errors='ignore')
        
        # Align columns between train and test
        features_df, test_features = self.align_columns(features_df, test_features)
        
        print(f"\nFinal training features shape: {features_df.shape}")
        if target_df is not None:
            print(f"Training target shape: {target_df.shape}")
        print(f"Test features shape: {test_features.shape}")
        
        return features_df, target_df, test_features
    
    def align_columns(self, train_df, test_df):
        """Align columns between train and test datasets"""
        # Get common columns
        common_cols = train_df.columns.intersection(test_df.columns)
        
        # Add missing columns with zeros
        for col in train_df.columns:
            if col not in test_df.columns:
                test_df[col] = 0
        
        for col in test_df.columns:
            if col not in train_df.columns:
                train_df[col] = 0
        
        # Reorder columns to match
        test_df = test_df[train_df.columns]
        
        return train_df, test_df
    
    def train_models(self, X, y):
        """Train multiple ML models and select the best one"""
        print("\n" + "="*60)
        print("MODEL TRAINING")
        print("="*60)
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Define models to try
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'XGBoost': XGBClassifier(random_state=42, n_estimators=100, 
                                     use_label_encoder=False, eval_metric='logloss')
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predict on validation set
            y_pred = model.predict(X_val_scaled)
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            roc_auc = roc_auc_score(y_val, y_pred_proba)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  ROC-AUC: {roc_auc:.4f}")
        
        # Select best model based on F1-score
        best_model_name = max(results, key=lambda x: results[x]['f1'])
        self.model = results[best_model_name]['model']
        
        print(f"\n🌟 Best Model: {best_model_name}")
        print(f"   F1-Score: {results[best_model_name]['f1']:.4f}")
        
        return results, X_val, y_val
    
    def evaluate_models(self, results, X_val, y_val):
        """Evaluate and compare all trained models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        # Create comparison table
        comparison_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[m]['accuracy'] for m in results],
            'Precision': [results[m]['precision'] for m in results],
            'Recall': [results[m]['recall'] for m in results],
            'F1-Score': [results[m]['f1'] for m in results],
            'ROC-AUC': [results[m]['roc_auc'] for m in results]
        })
        
        print("\nModel Performance Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Visualize results
        self.plot_model_comparison(comparison_df)
        self.plot_confusion_matrix(results, X_val, y_val)
        
        return comparison_df
    
    def plot_model_comparison(self, comparison_df):
        """Plot model performance comparison"""
        plt.figure(figsize=(12, 6))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(comparison_df['Model']))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            offset = width * (i - 1.5)
            plt.bar(x + offset, comparison_df[metric], width, label=metric)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, comparison_df['Model'], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, results, X_val, y_val):
        """Plot confusion matrix for best model"""
        best_model_name = max(results, key=lambda x: results[x]['f1'])
        y_pred = results[best_model_name]['predictions']
        
        cm = confusion_matrix(y_val, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Rejected', 'Approved'],
                   yticklabels=['Rejected', 'Approved'])
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self):
        """Save the trained model and preprocessing objects"""
        print("\n" + "="*60)
        print("SAVING MODEL")
        print("="*60)
        
        # Create directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save model
        model_path = 'models/loan_approval_model.pkl'
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
        
        # Save scaler
        scaler_path = 'models/scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
        
        # Save label encoders
        encoders_path = 'models/label_encoders.pkl'
        joblib.dump(self.label_encoders, encoders_path)
        print(f"Label encoders saved to {encoders_path}")
        
        # Save feature columns
        if self.feature_columns is not None:
            columns_path = 'models/feature_columns.pkl'
            joblib.dump(self.feature_columns, columns_path)
            print(f"Feature columns saved to {columns_path}")
        
        print("\n✅ All model artifacts saved successfully!")
    
    def make_predictions(self, test_features, test_df):
        """Make predictions on test data"""
        print("\n" + "="*60)
        print("MAKING PREDICTIONS ON TEST DATA")
        print("="*60)
        
        # Scale test features
        test_scaled = self.scaler.transform(test_features)
        
        # Make predictions
        predictions = self.model.predict(test_scaled)
        probabilities = self.model.predict_proba(test_scaled)[:, 1]
        
        # Create submission file (Kaggle format)
        submission_df = pd.DataFrame({
            'Loan_ID': test_df['Loan_ID'],
            'Loan_Status': ['Y' if p == 1 else 'N' for p in predictions],
            'Approval_Probability': probabilities
        })
        
        # Save submission file
        submission_df.to_csv('loan_predictions.csv', index=False)
        print(f"Predictions saved to 'loan_predictions.csv'")
        
        # Display prediction statistics
        print(f"\nPrediction Statistics:")
        print(f"Approved: {(predictions == 1).sum()} ({(predictions == 1).mean():.1%})")
        print(f"Rejected: {(predictions == 0).sum()} ({(predictions == 0).mean():.1%})")
        
        return submission_df

def main():
    """Main execution function"""
    print("🚀 AI Loan Approval System - Training with Kaggle Data")
    print("="*70)
    
    # Initialize model
    loan_model = LoanApprovalModel()
    
    # Step 1: Load data
    train_df, test_df = loan_model.load_data('train.csv', 'test.csv')
    
    # Step 2: Explore data
    loan_model.explore_data()
    
    # Step 3: Preprocess data
    X, y, test_features = loan_model.preprocess_data()
    loan_model.feature_columns = X.columns.tolist()
    
    # Step 4: Train models
    results, X_val, y_val = loan_model.train_models(X, y)
    
    # Step 5: Evaluate models
    comparison_df = loan_model.evaluate_models(results, X_val, y_val)
    
    # Step 6: Save model
    loan_model.save_model()
    
    # Step 7: Make predictions on test data
    submission_df = loan_model.make_predictions(test_features, test_df)
    
    print("\n" + "="*70)
    print("🎯 TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("1. models/loan_approval_model.pkl - Trained ML model")
    print("2. models/scaler.pkl - Feature scaler")
    print("3. models/label_encoders.pkl - Label encoders")
    print("4. loan_predictions.csv - Test predictions")
    print("5. model_comparison.png - Model performance chart")
    print("6. confusion_matrix.png - Confusion matrix")
    
    return loan_model

if __name__ == "__main__":
    main()