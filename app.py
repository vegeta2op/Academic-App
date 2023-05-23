

from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from flask import redirect, url_for
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import sqlite3
import csv

# Create a Flask application
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Initialize the SQLite3 database
def init_db():
    conn = sqlite3.connect('userpass.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, password TEXT)''')
    conn.commit()
    conn.close()

init_db()
@app.after_request
def add_cache_control(response):
    response.cache_control.no_cache = True
    return response

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["Email"]
        password = request.form["Password"]
        code = request.form["Code"]
        if code != "Zain":
            flash("Code not verified. Please try again.")
            return redirect("/")

        # Check if email is in the correct format
        # Check if email is in the correct format
        if not email.endswith("@amcec.edu"):
            error_messages = ["Invalid email address. Please use an email address from @amcec.edu domain."]
            return render_template("login.html", error_messages=error_messages)


        # Store the email and password in the database
        conn = sqlite3.connect('userpass.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (email, password))
            conn.commit()
            flash("You have successfully signed up!")
            return redirect("/")
        except sqlite3.IntegrityError:
            flash("Email already exists.")
        finally:
            conn.close()

    return render_template("login.html")


# Define a login page
def valid_user(email, password):
    conn = sqlite3.connect('userpass.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (email, password))
    user = c.fetchone()
    conn.close()
    return user is not None

def is_logged_in():
    return 'email' in session

@app.route("/", methods=["GET", "POST"])
def login():
    error_message = ""
    if request.method == "POST":
        email = request.form["Email"]
        password = request.form["Password"]

        if not email.endswith("@amcec.edu"):
            error_message = "Invalid email address. Please use an email address from @amcec.edu domain."
            return render_template("login.html", error_message=error_message)

        if not valid_user(email, password):
            error_message = "Invalid email or password"
        else:
            session['email'] = email
            return redirect('/dashboard')

    return render_template("login.html", error_message=error_message)


@app.route("/dashboard")
def dashboard():
    df = pd.read_csv('data.csv')
    if not is_logged_in():
        return redirect(url_for('login'))
    return render_template('dashboard.html',students=df.to_dict('records'))

# Load the student data from a CSV file
df = pd.read_csv('data.csv')

# Define a predictor page
from flask import request
import math

@app.route("/predictor")
def predictor():
    # Load the student data from a CSV file
    df = pd.read_csv('data.csv')
    # Render the predictor.html template with the student data
    return render_template('predictor.html', students=df.to_dict('records'))

# Define a view page for each student

# Define a check_marks page
def load_data():
    data = []
    with open('data.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

# Retrieve the marks for a specific student
def get_marks(usn):
    data = load_data()
    for row in data:
        if row['USN'] == usn:
            return row
    return None

# Define your routes and functions...
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from flask import jsonify



# Define a results page
@app.route('/view/<usn>', methods=['GET', 'POST'])
def results(usn):
    # Read the CSV file
    df = pd.read_csv('data.csv')

    # Filter the data to get the selected student
    student = df.loc[df['USN'] == int(usn)]

    # Get the sem1-sem5 values for the selected student
    sem1 = student['First sem'].values[0]
    sem2 = student['Second sem'].values[0]
    sem3 = student['Third sem'].values[0]
    sem4 = student['Fourth sem'].values[0]
    sem5 = student['Fifth sem'].values[0]
    # Prepare the input data for the Random Forest model
    X = df.iloc[:, 2:-1].values
    y = df.iloc[:, -1].values
    input_data = [[sem1, sem2, sem3, sem4, sem5]]
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train the Random Forest model using cross-validation
    model_cv = RandomForestRegressor(n_estimators=200, random_state=42)
    scores = cross_val_score(model_cv, X_train, y_train, cv=5, scoring='r2')
    r2_cv = scores.mean()
    # Train the Random model on the entire dataset
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    # Make a prediction for the selected student's marks
    predicted_marks = model.predict(input_data)
    # Test the model on the testing set
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    # Calculate the percentage increase or decrease in marks from the first semester to the last semester
    first_semester_marks = student.iloc[:, 2].values
    last_semester_marks = student.iloc[:, 7].values
    percentage_change = ((last_semester_marks - first_semester_marks) / first_semester_marks) * 100
    # Render the results.html template with the prediction results
    return render_template('view.html', student=student, usn=usn, sem1=sem1, sem2=sem2, sem3=sem3, sem4=sem4, sem5=sem5,predicted_marks=predicted_marks[0], r2_cv=r2_cv, r2=r2, mse=mse,percentage_change=percentage_change)


@app.route("/logout",methods=['GET', 'POST'])
def logout():
    session.pop('email', None)
    return redirect(url_for('login'))
if __name__ == '__main__':
    app.secret_key = 'your_secret_key'
    app.run(debug=True)
