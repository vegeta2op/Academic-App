from flask import Blueprint, render_template, request, redirect, url_for, session
from flask_login import login_required, logout_user
import pandas as pd
import csv
predictor_bp = Blueprint('students', __name__)

def is_logged_in():
    return 'email' in session


@predictor_bp.route("/predictor")
@login_required
def predictor():
    if not is_logged_in():
        return redirect(url_for('login'))
    # Load the student data from a CSV file
    df = pd.read_csv('data.csv')
    # Render the predictor.html template with the student data
    return render_template('predictor.html', students=df.to_dict('records'))

@predictor_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))
