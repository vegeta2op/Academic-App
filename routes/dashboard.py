from flask import Blueprint, render_template, request, redirect, url_for, session, after_this_request
from flask_login import login_required, logout_user
import pandas as pd
import csv
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from flask_wtf.csrf import CSRFProtect

dashboard_bp = Blueprint('dashboard', __name__)

def is_logged_in():
    return 'email' in session

class UpdateMarksForm(FlaskForm):
    usn = StringField('usn', validators=[DataRequired()])
    sem1 = StringField('sem1', validators=[DataRequired()])
    sem2 = StringField('sem2', validators=[DataRequired()])
    sem3 = StringField('sem3', validators=[DataRequired()])
    sem4 = StringField('sem4', validators=[DataRequired()])
    sem5 = StringField('sem5', validators=[DataRequired()])
    submit = SubmitField('submit')

@dashboard_bp.route("/dashboard")
@login_required  # Requires users to be logged in
def dashboard():
    if not is_logged_in():
        return redirect(url_for('login'))

    df = pd.read_csv('data.csv')
    email = session.get('email')  # Get the email from the session
    form = UpdateMarksForm()
    
    @after_this_request
    def add_header(response):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    
    return render_template('dashboard.html', students=df.to_dict('records'), form=form)

@dashboard_bp.route('/update-marks', methods=['POST'])
@login_required
def update_marks():
    if not is_logged_in():
        return redirect(url_for('login'))
    usn = request.form['usn']
    
    # Count the number of semester fields dynamically
    num_semesters = len([key for key in request.form.keys() if key.startswith('sem')])
    
    # Update the marks in the CSV file
    with open('data.csv', 'r') as file:
        students = list(csv.DictReader(file))
    
    fieldnames = students[0].keys() if students else []

    for student in students:
        if student['USN'] == usn:
            for i in range(1, num_semesters + 1):
                semester_key = f'sem{i}'
                if semester_key in request.form:
                    student[semester_key] = request.form[semester_key]

    with open('data.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(students)
    
    # Update df with the newly updated data
    df = pd.read_csv('data.csv')
    
    return render_template('dashboard.html', students=df.to_dict('records'), form=UpdateMarksForm())

@dashboard_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))