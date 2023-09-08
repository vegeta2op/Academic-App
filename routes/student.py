from flask import Blueprint, render_template, request, redirect, url_for,session
from flask_login import login_required , logout_user
import csv
import pandas as pd
from flask_wtf.csrf import CSRFProtect
student_bp = Blueprint('student', __name__)

def is_logged_in():
    return 'email' in session

@student_bp.route("/students")
@login_required
def student_route():
    if not is_logged_in():
        return redirect(url_for('login'))
    
    email = request.args.get('email')  # Get the email from the query parameter
    # Your students route logic here
    df = pd.read_csv('data.csv')
    return render_template('students.html',students=df.to_dict('records'))

@student_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))
df = pd.read_csv('data.csv')
@student_bp.route("/students")
def student():
    df = pd.read_csv('data.csv')

    return render_template('students.html', student=df.to_dict('records'))
