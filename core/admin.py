from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User
from .models import UserProfile, LoanApplication, CreditHistory, SystemLog

class UserProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False
    verbose_name_plural = 'Profile'

class CustomUserAdmin(UserAdmin):
    inlines = (UserProfileInline,)
    list_display = ('username', 'email', 'first_name', 'last_name', 'is_staff', 'date_joined')
    list_filter = ('is_staff', 'is_superuser', 'is_active', 'groups')
    search_fields = ('username', 'first_name', 'last_name', 'email')

@admin.register(LoanApplication)
class LoanApplicationAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'loan_type', 'loan_amount', 'status', 'predicted_status', 'applied_date')
    list_filter = ('status', 'loan_type', 'employment_type', 'marital_status', 'applied_date')
    search_fields = ('user__username', 'user__email', 'purpose')
    readonly_fields = ('applied_date', 'reviewed_date', 'decision_date')
    fieldsets = (
        ('Loan Details', {
            'fields': ('user', 'loan_type', 'loan_amount', 'tenure_months', 'interest_rate', 'purpose')
        }),
        ('Applicant Details', {
            'fields': ('age', 'income', 'employment_type', 'employment_years', 
                      'credit_score', 'marital_status', 'dependents')
        }),
        ('Financial Ratios', {
            'fields': ('debt_to_income_ratio', 'loan_to_value_ratio')
        }),
        ('Status & Decisions', {
            'fields': ('status', 'predicted_status', 'risk_score', 'approval_probability',
                      'applied_date', 'reviewed_date', 'decision_date')
        }),
        ('Existing Loans', {
            'fields': ('existing_loans', 'existing_loan_amount')
        }),
    )

@admin.register(CreditHistory)
class CreditHistoryAdmin(admin.ModelAdmin):
    list_display = ('user', 'payment_date', 'payment_amount', 'payment_status')
    list_filter = ('payment_status', 'payment_date')
    search_fields = ('user__username', 'remarks')

@admin.register(SystemLog)
class SystemLogAdmin(admin.ModelAdmin):
    list_display = ('log_type', 'message', 'user', 'ip_address', 'created_at')
    list_filter = ('log_type', 'created_at')
    search_fields = ('message', 'user__username')
    readonly_fields = ('created_at',)

# Re-register UserAdmin
admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)