🤖 AI Loan Approval System
A Django-based web application that uses machine learning to automate loan eligibility predictions in real-time.

![App Screenshot](https://github.com/ansarkp10/House_Price_Prediction_ML/blob/main/User_UI.png)

🚀 Quick Start
bash
# 1. Clone & setup
git clone https://github.com/ansarkp10/AI_Loan_Approval_System.git
cd AI_Loan_Approval_System/loan_system

# 2. Install & run
python -m venv venv
venv\Scripts\activate  # Windows
pip install django scikit-learn pandas numpy
python manage.py migrate
python manage.py runserver
Visit http://127.0.0.1:8000

✨ Features
Instant AI Prediction – ML model evaluates applications in real-time

Risk Scoring – Generates risk scores (0-100%) and approval probabilities

Admin Dashboard – Manage applications and override decisions

Transparent Analysis – Shows key factors affecting decisions

🏗️ Tech Stack
Backend: Django, Python

ML: Scikit-learn, Pandas

Frontend: HTML, CSS, Bootstrap

Database: SQLite

📁 Project Structure
text
loan_system/
├── core/              # Main app (models, views, ML logic)
├── templates/         # HTML pages
├── data/              # Training datasets
└── manage.py          # Django starter
📌 Key Files
core/ml_model/train_model.py – ML model training

templates/apply_loan.html – Application form

templates/admin_dashboard.html – Admin panel

🔗 Links
GitHub: github.com/ansarkp10/AI_Loan_Approval_System

Technologies: Django, Scikit-learn, Bootstrap
