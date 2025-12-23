from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    phone_number = models.CharField(max_length=15)
    address = models.TextField()
    date_of_birth = models.DateField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username}'s Profile"

class LoanApplication(models.Model):
    LOAN_TYPES = [
        ('personal', 'Personal Loan'),
        ('home', 'Home Loan'),
        ('car', 'Car Loan'),
        ('education', 'Education Loan'),
        ('business', 'Business Loan'),
    ]
    
    EMPLOYMENT_TYPES = [
        ('salaried', 'Salaried'),
        ('self_employed', 'Self Employed'),
        ('business', 'Business'),
        ('unemployed', 'Unemployed'),
    ]
    
    MARITAL_STATUS = [
        ('single', 'Single'),
        ('married', 'Married'),
        ('divorced', 'Divorced'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    loan_type = models.CharField(max_length=20, choices=LOAN_TYPES)
    loan_amount = models.DecimalField(max_digits=12, decimal_places=2, validators=[MinValueValidator(1000)])
    tenure_months = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(360)])
    interest_rate = models.DecimalField(max_digits=4, decimal_places=2, default=8.5)
    
    # Applicant Details
    age = models.IntegerField()
    income = models.DecimalField(max_digits=10, decimal_places=2)
    employment_type = models.CharField(max_length=20, choices=EMPLOYMENT_TYPES)
    employment_years = models.IntegerField(default=0)
    credit_score = models.IntegerField(default=300, validators=[MinValueValidator(300), MaxValueValidator(850)])
    marital_status = models.CharField(max_length=10, choices=MARITAL_STATUS)
    dependents = models.IntegerField(default=0)
    
    # ML Features
    debt_to_income_ratio = models.DecimalField(max_digits=5, decimal_places=2, default=0.0)
    loan_to_value_ratio = models.DecimalField(max_digits=5, decimal_places=2, default=0.0)
    
    # Application Status
    status = models.CharField(max_length=20, default='pending', choices=[
        ('pending', 'Pending'),
        ('approved', 'Approved'),
        ('rejected', 'Rejected'),
        ('under_review', 'Under Review'),
    ])
    
    # ML Prediction Results
    predicted_status = models.CharField(max_length=20, blank=True)
    risk_score = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    approval_probability = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    
    # Timestamps
    applied_date = models.DateTimeField(auto_now_add=True)
    reviewed_date = models.DateTimeField(null=True, blank=True)
    decision_date = models.DateTimeField(null=True, blank=True)
    
    # Additional Info
    purpose = models.TextField(blank=True)
    existing_loans = models.IntegerField(default=0)
    existing_loan_amount = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    
    def __str__(self):
        return f"{self.user.username} - {self.loan_type} - {self.status}"
    
    @property
    def monthly_installment(self):
        if self.interest_rate and self.tenure_months and self.loan_amount:
            # Convert Decimal to float for calculation
            r = float(self.interest_rate) / 100 / 12
            n = self.tenure_months
            p = float(self.loan_amount)
            
            # EMI formula: EMI = [P x R x (1+R)^N]/[(1+R)^N-1]
            emi = p * r * ((1 + r) ** n) / (((1 + r) ** n) - 1)
            return round(emi, 2)
        return 0
    
    class Meta:
        ordering = ['-applied_date']

class CreditHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    loan_application = models.ForeignKey(LoanApplication, on_delete=models.CASCADE, null=True, blank=True)
    payment_date = models.DateField()
    payment_amount = models.DecimalField(max_digits=10, decimal_places=2)
    payment_status = models.CharField(max_length=20, choices=[
        ('ontime', 'On Time'),
        ('late', 'Late'),
        ('defaulted', 'Defaulted'),
    ])
    remarks = models.TextField(blank=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.payment_date} - {self.payment_status}"
    
class SystemLog(models.Model):
    LOG_TYPES = [
        ('application', 'Application'),
        ('prediction', 'Prediction'),
        ('system', 'System'),
        ('error', 'Error'),
    ]
    
    log_type = models.CharField(max_length=20, choices=LOG_TYPES)
    message = models.TextField()
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.log_type} - {self.message[:50]}"