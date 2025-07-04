import os
import logging
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect, validate_csrf
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from flask_session import Session
from wtforms import StringField, PasswordField, SubmitField, validators
from wtforms.validators import DataRequired, Length, EqualTo, Email
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.middleware.proxy_fix import ProxyFix
from config import config
import secrets

# Initialize Flask app
app = Flask(__name__)

# Load configuration
config_name = os.getenv('FLASK_ENV', 'default')
app.config.from_object(config[config_name])

# Security enhancements
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
csrf = CSRFProtect(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'
Session(app)
CORS(app, origins=['http://localhost:3000'])  # Adjust for your frontend

# Initialize Flask-Limiter with memory storage
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",  # Use memory storage instead of Redis
    strategy="fixed-window"
)

# Logging setup
logging.basicConfig(
    level=getattr(logging, app.config.get('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database Models
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(60), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    failed_login_attempts = db.Column(db.Integer, default=0)
    locked_until = db.Column(db.DateTime)
    
    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    
    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)
    
    def is_locked(self):
        if self.locked_until and datetime.utcnow() < self.locked_until:
            return True
        return False

class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    action = db.Column(db.String(100), nullable=False)
    details = db.Column(db.Text)
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Forms with enhanced validation
class RegistrationForm(FlaskForm):
    email = StringField('Email Address', validators=[
        DataRequired(),
        Email(message='Invalid email address'),
        Length(min=6, max=120)
    ])
    password = PasswordField('Password', validators=[
        DataRequired(),
        Length(min=8, max=128, message="Password must be between 8-128 characters"),
        validators.Regexp(
            r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]',
            message="Password must contain at least one uppercase letter, one lowercase letter, one digit, and one special character"
        )
    ])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(),
        EqualTo('password', message='Passwords must match')
    ])
    submit = SubmitField('Sign Up')

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

# Utility functions
def log_action(action, details=None, user_id=None):
    """Log user actions for audit purposes"""
    try:
        log_entry = AuditLog(
            user_id=user_id or (current_user.id if current_user.is_authenticated else None),
            action=action,
            details=details,
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent')
        )
        db.session.add(log_entry)
        db.session.commit()
        logger.info(f"Action logged: {action} for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to log action: {e}")

def validate_csrf_token():
    """Validate CSRF token for AJAX requests"""
    try:
        validate_csrf(request.headers.get('X-CSRFToken'))
        return True
    except Exception:
        return False

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error_code=404, error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('error.html', error_code=500, error_message="Internal server error"), 500

@app.errorhandler(403)
def forbidden_error(error):
    return render_template('error.html', error_code=403, error_message="Access forbidden"), 403

@app.errorhandler(429)
def ratelimit_handler(e):
    return render_template('error.html', error_code=429, error_message="Rate limit exceeded"), 429

# Routes
@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data.lower()).first()
        
        if user and user.is_locked():
            flash('Account is temporarily locked due to too many failed login attempts.', 'danger')
            log_action('login_attempt_locked', f'Email: {form.email.data}')
            return render_template('login.html', form=form, registration_form=RegistrationForm())
        
        if user and user.check_password(form.password.data):
            # Reset failed attempts on successful login
            user.failed_login_attempts = 0
            user.locked_until = None
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            login_user(user)
            session['email'] = user.email
            log_action('login_success', user_id=user.id)
            
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard.dashboard'))
        else:
            if user:
                user.failed_login_attempts += 1
                if user.failed_login_attempts >= 5:
                    user.locked_until = datetime.utcnow() + timedelta(minutes=30)
                db.session.commit()
            
            flash('Invalid email or password', 'danger')
            log_action('login_failure', f'Email: {form.email.data}')
    
    return render_template('login.html', form=form, registration_form=RegistrationForm())

@app.route('/register', methods=['POST'])
@app.route('/signup', methods=['POST'])
@limiter.limit("3 per minute")
def signup():
    form = RegistrationForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(email=form.email.data.lower()).first()
        if existing_user:
            flash('Email already registered', 'danger')
            log_action('signup_attempt_duplicate', f'Email: {form.email.data}')
            return redirect(url_for('login'))
        
        user = User(email=form.email.data.lower())
        user.set_password(form.password.data)
        
        try:
            db.session.add(user)
            db.session.commit()
            login_user(user)
            session['email'] = user.email
            log_action('signup_success', user_id=user.id)
            flash('Registration successful!', 'success')
            return redirect(url_for('dashboard.dashboard'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Registration error: {e}")
            flash('Registration failed. Please try again.', 'danger')
    
    # If validation fails, show errors
    for field, errors in form.errors.items():
        for error in errors:
            flash(f'{field}: {error}', 'danger')
    
    return redirect(url_for('login'))

@app.route('/logout')
@login_required
def logout():
    log_action('logout', user_id=current_user.id)
    logout_user()
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

# API Routes for modern frontend interaction
@app.route('/api/csrf-token')
def get_csrf_token():
    """Provide CSRF token for AJAX requests"""
    return jsonify({'csrf_token': csrf.generate_csrf()})

@app.route('/api/user/profile')
@login_required
def get_user_profile():
    """Get current user profile"""
    return jsonify({
        'id': current_user.id,
        'email': current_user.email,
        'created_at': current_user.created_at.isoformat(),
        'last_login': current_user.last_login.isoformat() if current_user.last_login else None
    })

# Health check endpoint
@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '2.0.0'
    })

# Security headers
@app.after_request
def after_request(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com;"
    return response

# Register blueprints
from routes.dashboard import dashboard_bp
from routes.student import student_bp
from routes.predictor import predictor_bp
from routes.view import view_bp
from routes.analytics import analytics_bp
from routes.reports import reports_bp
from routes.settings import settings_bp

app.register_blueprint(dashboard_bp)
app.register_blueprint(student_bp)
app.register_blueprint(predictor_bp)
app.register_blueprint(view_bp)
app.register_blueprint(analytics_bp)
app.register_blueprint(reports_bp)
app.register_blueprint(settings_bp)

# Database initialization
def init_db():
    """Initialize database with tables"""
    with app.app_context():
        db.create_all()
        logger.info("Database initialized")

if __name__ == '__main__':
    init_db()
    port = int(os.environ.get('PORT', 8081))
    app.run(host='0.0.0.0', port=port, debug=app.config.get('DEBUG', False))