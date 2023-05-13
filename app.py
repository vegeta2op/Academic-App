

from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import sqlite3

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

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["Email"]
        password = request.form["Password"]
        code = request.form["Code"]
        if code != "Zain":
            flash("Code not verified. Please try again.")
            return redirect("/")

        # Store the email and password in the database
        conn = sqlite3.connect('userpass.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, password))
            conn.commit()
            flash("You have successfully signed up!")
            return redirect("/")
        except sqlite3.IntegrityError:
            flash("Email already exists.")
        finally:
            conn.close()

    return render_template("signup.html")

# Define a login page
def valid_user(email, password):
            conn = sqlite3.connect('userpass.db')
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (email, password))
            user = c.fetchone()
            conn.close()
            return user is not None
        # Check if the email and password match the stored values
@app.route("/", methods=["GET", "POST"])
def login():
    error_message = ""
    if request.method == "POST":
        email = request.form["Email"]
        password = request.form["Password"]


        if not valid_user(email, password):
            error_message = 'Invalid email or password'

        else:
            return redirect('/dashboard')


    return render_template("login.html", error_message=error_message)

@app.route("/dashboard")
def dashboard():
    return render_template('dashboard.html')

# Load the student data from a CSV file
df = pd.read_csv('data.csv')

# Define a predictor page
@app.route("/predictor")
def predictor():
    # Load the student data from a CSV file
    df = pd.read_csv('data.csv')
    # Render the predictor.html template with the student data
    return render_template('predictor.html', students=df.to_dict('records'))

# Define a view page for each student
@app.route("/view/<usn>")
def view(usn):

    # Filter the data to get the selected student
    student = df.loc[df['USN'] == usn]

    # Get the USN and name of the student
    usn = student['USN']
    name = student['Name']

    # Render the view.html template with the selected student data
    return render_template('view.html', student=student, usn=usn, name=name)

# Define a check_marks page
@app.route('/check_marks', methods=['POST'])
def check_marks():
    sem1 = int(request.form['sem1'])
    sem2 = int(request.form['sem2'])
    sem3 = int(request.form['sem3'])
    sem4 = int(request.form['sem4'])
    sem5 = int(request.form['sem5'])
    usn = request.form['usn']

    if sem1 < 35 or sem2 < 35 or sem3 < 35 or sem4 < 35 or sem5 < 35:
        return render_template('ineligible.html')
    else:
        session['usn'] = usn
        session['sem1'] = sem1
        session['sem2'] = sem2
        session['sem3'] = sem3
        session['sem4'] = sem4
        session['sem5'] = sem5
        return redirect('/results')


# Define a results page
@app.route('/results', methods=['GET', 'POST'])
def results():

    usn = session.get('usn')
    student = df.loc[df['USN'] == usn]
    sem1 = session.get('sem1')
    sem2 = session.get('sem2')
    sem3 = session.get('sem3')
    sem4 = session.get('sem4')
    sem5 = session.get('sem5')

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
    return render_template('results.html', student=student, predicted_marks=predicted_marks[0], r2_cv=r2_cv, r2=r2, mse=mse, percentage_change=percentage_change)

if __name__ == '__main__':
    app.secret_key = 'your_secret_key'
    app.run(debug=True)
