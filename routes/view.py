from flask import Blueprint, render_template, request, redirect, url_for,session
from flask_login import login_required , logout_user
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Lasso, LassoCV
import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns
view_bp=Blueprint('view/<usn>',__name__)

def is_logged_in():
    return 'email' in session

@view_bp.route('/view/<usn>', methods=['GET', 'POST'])
@login_required
def results(usn):
    if not is_logged_in():
        return redirect(url_for('login'))
    df = pd.read_csv('data.csv')

    
    student = df.loc[df['USN'] == int(usn)]

    # Get the sem1-sem5 values for the selected student
    sem1 = student['sem1'].values[0]
    sem2 = student['sem2'].values[0]
    sem3 = student['sem3'].values[0]
    sem4 = student['sem4'].values[0]
    sem5 = student['sem5'].values[0]
    # Prepare the input data for the Random Forest model
    X = df.iloc[:, 2:-1].values
    y = df.iloc[:, -1].values
    input_data = [[sem1, sem2, sem3, sem4, sem5]]
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train the Lasso Regression model using cross-validation
    model_cv = LassoCV(cv=5, random_state=42)
    model_cv.fit(X_train, y_train)
    r2_cv = model_cv.score(X_train, y_train)
    # Train the Lasso model on the entire dataset
    model = Lasso(alpha=1, random_state=42)
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

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel('Actual Marks')
    plt.ylabel('Predicted Marks')
    plt.title('Actual vs. Predicted Marks')
    graph_filename = 'scatter_plot.png'
    plt.savefig(os.path.join('static', graph_filename))  # Save the plot in the 'static' folder

    # Render the results.html template with the prediction results, graph name, and other variables
    return render_template('view.html', student=student,usn=usn, sem1=sem1, sem2=sem2, sem3=sem3, sem4=sem4, sem5=sem5 ,predicted_marks=predicted_marks[0], r2_cv=r2_cv, r2=r2, mse=mse, percentage_change=percentage_change, graph_filename=graph_filename)