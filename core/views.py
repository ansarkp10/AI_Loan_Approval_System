from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.admin.views.decorators import staff_member_required
from django.db.models import Count, Avg, Sum
from django.http import JsonResponse
from .forms import UserRegistrationForm, UserLoginForm, LoanApplicationForm
from .models import LoanApplication, UserProfile, SystemLog, CreditHistory
from .ml_model.train_model import predict_loan_approval
import json
from datetime import datetime, timedelta

def home(request):
    """Home page view"""
    return render(request, 'home.html')

def register(request):
    """User registration view"""
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            
            # Create user profile
            UserProfile.objects.create(
                user=user,
                phone_number=form.cleaned_data['phone_number'],
                address=form.cleaned_data['address'],
                date_of_birth=form.cleaned_data['date_of_birth']
            )
            
            # Log system event
            SystemLog.objects.create(
                log_type='system',
                message=f'New user registered: {user.username}',
                user=user
            )
            
            messages.success(request, 'Registration successful! Please login.')
            return redirect('login')
    else:
        form = UserRegistrationForm()
    
    return render(request, 'register.html', {'form': form})

def user_login(request):
    """User login view"""
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    if request.method == 'POST':
        form = UserLoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            
            if user:
                login(request, user)
                
                # Log system event
                SystemLog.objects.create(
                    log_type='system',
                    message=f'User logged in: {user.username}',
                    user=user,
                    ip_address=request.META.get('REMOTE_ADDR')
                )
                
                messages.success(request, 'Login successful!')
                return redirect('dashboard')
            else:
                messages.error(request, 'Invalid credentials')
    else:
        form = UserLoginForm()
    
    return render(request, 'login.html', {'form': form})

@login_required
def user_logout(request):
    """User logout view"""
    logout(request)
    messages.success(request, 'Logged out successfully!')
    return redirect('home')

@login_required
def dashboard(request):
    """User dashboard view"""
    user_applications = LoanApplication.objects.filter(user=request.user).order_by('-applied_date')
    
    # Get statistics
    total_applications = user_applications.count()
    approved_applications = user_applications.filter(status='approved').count()
    pending_applications = user_applications.filter(status='pending').count()
    
    context = {
        'user_applications': user_applications,
        'total_applications': total_applications,
        'approved_applications': approved_applications,
        'pending_applications': pending_applications,
    }
    
    return render(request, 'dashboard.html', context)

# core/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import LoanApplicationForm
from .models import LoanApplication

# Try to import ML predictor, but don't crash if not available
try:
    from .ml_integration import get_predictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("ML integration module not available")

@login_required
def apply_loan(request):
    """Handle loan application"""
    if request.method == 'POST':
        form = LoanApplicationForm(request.POST)
        if form.is_valid():
            loan_application = form.save(commit=False)
            loan_application.user = request.user
            loan_application.interest_rate = 8.5  # Default interest rate
            
            # GET CALCULATED RATIOS FROM FORM CLEAN METHOD
            debt_to_income_ratio = form.cleaned_data.get('debt_to_income_ratio', 0.0)
            loan_to_value_ratio = form.cleaned_data.get('loan_to_value_ratio', 0.0)
            
            # SET THE CALCULATED RATIOS ON THE MODEL
            loan_application.debt_to_income_ratio = debt_to_income_ratio
            loan_application.loan_to_value_ratio = loan_to_value_ratio
            
            # Prepare data for ML prediction (if available)
            ml_data = {
                'age': loan_application.age,
                'income': float(loan_application.income),
                'employment_type': loan_application.employment_type,
                'employment_years': loan_application.employment_years,
                'credit_score': loan_application.credit_score,
                'marital_status': loan_application.marital_status,
                'dependents': loan_application.dependents,
                'loan_amount': float(loan_application.loan_amount),
                'loan_type': loan_application.loan_type,
                'tenure_months': loan_application.tenure_months,
                'debt_to_income_ratio': float(debt_to_income_ratio),
                'loan_to_value_ratio': float(loan_to_value_ratio),
                'existing_loans': loan_application.existing_loans,
                'existing_loan_amount': float(loan_application.existing_loan_amount),
            }
            
            # Get ML prediction if available
            if ML_AVAILABLE:
                try:
                    predictor = get_predictor()
                    prediction = predictor.predict(ml_data)
                    
                    loan_application.predicted_status = prediction['status']
                    loan_application.risk_score = prediction['risk_score']
                    loan_application.approval_probability = prediction['probability'] * 100
                    
                    if prediction['approved']:
                        loan_application.status = 'under_review'
                    else:
                        loan_application.status = 'rejected'
                        
                except Exception as e:
                    print(f"ML prediction failed: {e}")
                    loan_application.status = 'pending'
                    messages.warning(request, 'AI prediction service is temporarily unavailable.')
            else:
                loan_application.status = 'pending'
                messages.info(request, 'AI model not loaded. Application saved for manual review.')
            
            # SAVE THE APPLICATION WITH ALL FIELDS
            loan_application.save()
            
            messages.success(request, 'Loan application submitted successfully!')
            return redirect('loan_status', loan_id=loan_application.id)
    else:
        form = LoanApplicationForm()
    
    return render(request, 'apply_loan.html', {
        'form': form,
        'ml_available': ML_AVAILABLE
    })

@login_required
def loan_status(request, loan_id):
    """View loan application status"""
    loan_application = get_object_or_404(LoanApplication, id=loan_id, user=request.user)
    
    context = {
        'loan': loan_application,
        'emi': loan_application.monthly_installment,
    }
    
    return render(request, 'loan_status.html', context)

@staff_member_required
def admin_dashboard(request):
    """Admin dashboard view"""
    # Get statistics
    total_applications = LoanApplication.objects.count()
    approved_applications = LoanApplication.objects.filter(status='approved').count()
    pending_applications = LoanApplication.objects.filter(status='pending').count()
    rejected_applications = LoanApplication.objects.filter(status='rejected').count()
    
    # Get recent applications
    recent_applications = LoanApplication.objects.all().order_by('-applied_date')[:10]
    
    # Get statistics by loan type
    loan_type_stats = LoanApplication.objects.values('loan_type').annotate(
        count=Count('id'),
        approval_rate=Avg('approval_probability')
    )
    
    # Get daily applications for last 7 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    daily_applications = []
    for i in range(7):
        date = start_date + timedelta(days=i)
        count = LoanApplication.objects.filter(
            applied_date__date=date.date()
        ).count()
        daily_applications.append({
            'date': date.strftime('%Y-%m-%d'),
            'count': count
        })
    
    context = {
        'total_applications': total_applications,
        'approved_applications': approved_applications,
        'pending_applications': pending_applications,
        'rejected_applications': rejected_applications,
        'recent_applications': recent_applications,
        'loan_type_stats': loan_type_stats,
        'daily_applications': json.dumps(daily_applications),
    }
    
    return render(request, 'admin_dashboard.html', context)

@staff_member_required
def update_loan_status(request, loan_id):
    """Update loan status (admin only)"""
    if request.method == 'POST':
        loan = get_object_or_404(LoanApplication, id=loan_id)
        new_status = request.POST.get('status')
        
        if new_status in ['approved', 'rejected', 'under_review']:
            loan.status = new_status
            loan.decision_date = datetime.now()
            loan.save()
            
            # Log status change
            SystemLog.objects.create(
                log_type='application',
                message=f'Loan status updated: {loan.id} -> {new_status}',
                user=request.user
            )
            
            messages.success(request, 'Loan status updated successfully!')
    
    return redirect('admin_dashboard')

def get_prediction_analytics(request):
    """Get ML model analytics"""
    if not request.user.is_staff:
        return JsonResponse({'error': 'Unauthorized'}, status=403)
    
    # Calculate prediction accuracy
    applications = LoanApplication.objects.filter(
        predicted_status__isnull=False,
        status__in=['approved', 'rejected']
    )
    
    if applications.exists():
        correct_predictions = 0
        total = 0
        
        for app in applications:
            if (app.predicted_status == 'Approved' and app.status == 'approved') or \
               (app.predicted_status == 'Rejected' and app.status == 'rejected'):
                correct_predictions += 1
            total += 1
        
        accuracy = (correct_predictions / total * 100) if total > 0 else 0
        
        return JsonResponse({
            'accuracy': round(accuracy, 2),
            'total_predictions': total,
            'correct_predictions': correct_predictions
        })
    
    return JsonResponse({'message': 'No prediction data available'})