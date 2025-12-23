from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import UserProfile, LoanApplication
from django.core.validators import MinValueValidator

class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    first_name = forms.CharField(max_length=30, required=True)
    last_name = forms.CharField(max_length=30, required=True)
    phone_number = forms.CharField(max_length=15, required=True)
    address = forms.CharField(widget=forms.Textarea(attrs={'rows': 3}), required=True)
    date_of_birth = forms.DateField(
        widget=forms.DateInput(attrs={'type': 'date'}),
        required=True
    )

    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'last_name', 'password1', 'password2']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Remove ALL help text from ALL fields
        for field in self.fields.values():
            field.help_text = None
        
        # Remove labels if you want just placeholder text
        self.fields['username'].label = ''
        self.fields['email'].label = ''
        self.fields['first_name'].label = ''
        self.fields['last_name'].label = ''
        self.fields['password1'].label = ''
        self.fields['password2'].label = ''
        self.fields['phone_number'].label = ''
        self.fields['address'].label = ''
        self.fields['date_of_birth'].label = ''
        
        # Add placeholder text instead
        self.fields['username'].widget.attrs.update({'placeholder': 'Username'})
        self.fields['email'].widget.attrs.update({'placeholder': 'Email'})
        self.fields['first_name'].widget.attrs.update({'placeholder': 'First Name'})
        self.fields['last_name'].widget.attrs.update({'placeholder': 'Last Name'})
        self.fields['password1'].widget.attrs.update({'placeholder': 'Password'})
        self.fields['password2'].widget.attrs.update({'placeholder': 'Confirm Password'})
        self.fields['phone_number'].widget.attrs.update({'placeholder': 'Phone Number'})
        self.fields['address'].widget.attrs.update({'placeholder': 'Address'})
        self.fields['date_of_birth'].widget.attrs.update({'placeholder': 'Date of Birth'})

class UserLoginForm(forms.Form):
    username = forms.CharField(
        max_length=150,
        widget=forms.TextInput(attrs={'placeholder': 'Username'}),
        label=''
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'placeholder': 'Password'}),
        label=''
    )

class LoanApplicationForm(forms.ModelForm):
    class Meta:
        model = LoanApplication
        fields = [
            'loan_type', 'loan_amount', 'tenure_months', 'purpose',
            'age', 'income', 'employment_type', 'employment_years',
            'credit_score', 'marital_status', 'dependents',
            'existing_loans', 'existing_loan_amount'
        ]
        widgets = {
            'purpose': forms.Textarea(attrs={'rows': 3}),
            'loan_amount': forms.NumberInput(attrs={'min': 1000, 'step': 1000}),
            'tenure_months': forms.NumberInput(attrs={'min': 1, 'max': 360}),
            'age': forms.NumberInput(attrs={'min': 18, 'max': 70}),
            'income': forms.NumberInput(attrs={'min': 0, 'step': 1000}),
            'employment_years': forms.NumberInput(attrs={'min': 0}),
            'credit_score': forms.NumberInput(attrs={'min': 300, 'max': 850}),
            'dependents': forms.NumberInput(attrs={'min': 0}),
            'existing_loans': forms.NumberInput(attrs={'min': 0}),
            'existing_loan_amount': forms.NumberInput(attrs={'min': 0}),
        }
    
    def clean(self):
        cleaned_data = super().clean()
        
        # Get values with defaults
        loan_amount = cleaned_data.get('loan_amount', 0)
        income = cleaned_data.get('income', 0)
        existing_loan_amount = cleaned_data.get('existing_loan_amount', 0)
        
        # Calculate debt-to-income ratio
        if income and float(income) > 0:
            total_debt = float(existing_loan_amount) + float(loan_amount)
            debt_to_income_ratio = total_debt / float(income)
        else:
            debt_to_income_ratio = 0.0
        
        cleaned_data['debt_to_income_ratio'] = debt_to_income_ratio
        
        # Calculate loan-to-value ratio (simplified)
        if loan_amount and float(loan_amount) > 0:
            property_value = float(loan_amount) * 1.2  # Assuming 20% higher
            loan_to_value_ratio = float(loan_amount) / property_value
        else:
            loan_to_value_ratio = 0.0
        
        cleaned_data['loan_to_value_ratio'] = loan_to_value_ratio
        
        return cleaned_data