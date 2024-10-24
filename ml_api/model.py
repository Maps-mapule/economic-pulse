from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class LoanApplication(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer, nullable=False)
    job = db.Column(db.String(50), nullable=False)
    marital = db.Column(db.String(20), nullable=False)
    education = db.Column(db.String(50), nullable=False)
    default = db.Column(db.String(5), nullable=False)
    housing = db.Column(db.String(5), nullable=False)
    loan = db.Column(db.String(5), nullable=False)
    contact = db.Column(db.String(20), nullable=False)
    month = db.Column(db.String(20), nullable=False)
    day_of_week = db.Column(db.String(20), nullable=False)
    duration = db.Column(db.Integer, nullable=False)
    campaign = db.Column(db.Integer, nullable=False)
    pdays = db.Column(db.Integer, nullable=False)
    previous = db.Column(db.Integer, nullable=False)
    poutcome = db.Column(db.String(20), nullable=False)
    emp_var_rate = db.Column(db.Float, nullable=False)
    cons_price_idx = db.Column(db.Float, nullable=False)
    cons_conf_idx = db.Column(db.Float, nullable=False)
    euribor3m = db.Column(db.Float, nullable=False)
    nr_employed = db.Column(db.Float, nullable=False)
    loan_application = db.Column(db.Boolean, nullable=False)
