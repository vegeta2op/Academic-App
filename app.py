from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField,validators
from wtforms.validators import DataRequired, Length, EqualTo
import os
from flask_wtf.csrf import CSRFProtect
from flask import Flask, session
from flask_session import Session




app = Flask(__name__)
csrf = CSRFProtect(app)
app.config['SECRET_KEY'] = '3893298h88932@298h23;32h'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///userpass.db'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

from routes.dashboard import dashboard_bp 
from routes.student import student_bp 
from routes.predictor import predictor_bp
from routes.view import view_bp


# Define the User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class RegistrationForm(FlaskForm):
    email = StringField('Email Address', validators=[validators.DataRequired(), validators.Email()])
    password = PasswordField('Password', validators=[
        validators.DataRequired(),
        validators.Length(min=8, message="Password must be at least 8 characters long")
    ])
    re_password = PasswordField('Confirm Password', validators=[
        validators.DataRequired(),
        validators.EqualTo('password', message='Passwords must match')
    ])
    submit = SubmitField('Signup')



class LoginForm(FlaskForm):
    email = StringField('email', validators=[DataRequired(), Length(min=6, max=120)])
    password = PasswordField('password', validators=[DataRequired()])
    submit = SubmitField('login-btn')




# Route for handling the signup form submission
@app.route("/signup", methods=["POST"])
def signup():
    form = RegistrationForm(request.form)
    if form.validate():
        # Handle the signup form submission
        email = form.email.data
        password = form.password.data
        confirm_password = form.re_password.data

        if password != confirm_password:
            error_message = "Passwords must match"
            return redirect(url_for("home"))
        else:
            user = User.query.filter_by(email=email).first()
            if user:
                error_message = "Email already exists"
                return redirect(url_for("home"))
            else:
                hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
                new_user = User(email=email, password=hashed_password)
                db.session.add(new_user)
                db.session.commit()
                login_user(new_user)
                return redirect('/dashboard')  # Redirect to your dashboard or desired page

# Route for handling the login form submission
@app.route("/", methods=["GET", "POST"])
def login():
    error_message=""
    if request.method == "POST":
        form = LoginForm(request.form)
        if form.validate_on_submit():
            print(form.email.data)
            user = User.query.filter_by(email=form.email.data).first()
            if user and bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                session['email'] = user.email
                return redirect('/dashboard')  
            else:
                error_message = "Invalid email or password"
        return render_template("login.html", registration_form=RegistrationForm(), login_form=form, error_message="Invalid email or password")
    else:
        # If it's a GET request, just render the login page
        return render_template("login.html", registration_form=RegistrationForm(), login_form=LoginForm())

app.register_blueprint(predictor_bp)
app.register_blueprint(dashboard_bp)
app.register_blueprint(student_bp)
app.register_blueprint(view_bp)
if __name__ == '__main__':
    # Ensure you create database tables after the app context is set up
    with app.app_context():
        db.create_all()
    app.run(debug=True)