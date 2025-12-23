from django.urls import path
from . import views

urlpatterns = [
    # Authentication URLs
    path('', views.home, name='home'),
    path('register/', views.register, name='register'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    
    # User URLs
    path('dashboard/', views.dashboard, name='dashboard'),
    path('apply-loan/', views.apply_loan, name='apply_loan'),
    path('loan-status/<int:loan_id>/', views.loan_status, name='loan_status'),
    
    # Admin URLs
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('update-loan-status/<int:loan_id>/', views.update_loan_status, name='update_loan_status'),
    
    # API URLs
    path('api/prediction-analytics/', views.get_prediction_analytics, name='prediction_analytics'),
]