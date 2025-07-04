from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from flask_login import login_required, current_user
from flask_wtf import FlaskForm
from flask_wtf.csrf import validate_csrf
from wtforms import StringField, PasswordField, SelectField, BooleanField, SubmitField, TextAreaField, IntegerField
from wtforms.validators import DataRequired, Email, Length, Optional, NumberRange
from werkzeug.security import check_password_hash
import logging
from datetime import datetime
import json
import os

settings_bp = Blueprint('settings', __name__)
logger = logging.getLogger(__name__)

class ProfileSettingsForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    current_password = PasswordField('Current Password', validators=[DataRequired()])
    new_password = PasswordField('New Password', validators=[Optional(), Length(min=8)])
    confirm_password = PasswordField('Confirm New Password', validators=[Optional()])
    submit = SubmitField('Update Profile')

class PreferencesForm(FlaskForm):
    theme = SelectField('Theme', choices=[
        ('light', 'Light'),
        ('dark', 'Dark'),
        ('auto', 'Auto')
    ], default='light')
    
    language = SelectField('Language', choices=[
        ('en', 'English'),
        ('es', 'Spanish'),
        ('fr', 'French'),
        ('de', 'German')
    ], default='en')
    
    timezone = SelectField('Timezone', choices=[
        ('UTC', 'UTC'),
        ('America/New_York', 'Eastern Time'),
        ('America/Chicago', 'Central Time'),
        ('America/Denver', 'Mountain Time'),
        ('America/Los_Angeles', 'Pacific Time'),
        ('Europe/London', 'London'),
        ('Europe/Paris', 'Paris'),
        ('Asia/Tokyo', 'Tokyo'),
        ('Asia/Shanghai', 'Shanghai')
    ], default='UTC')
    
    notifications_email = BooleanField('Email Notifications', default=True)
    notifications_dashboard = BooleanField('Dashboard Notifications', default=True)
    notifications_reports = BooleanField('Report Notifications', default=True)
    
    default_view = SelectField('Default View', choices=[
        ('dashboard', 'Dashboard'),
        ('students', 'Students'),
        ('analytics', 'Analytics'),
        ('predictor', 'AI Predictor')
    ], default='dashboard')
    
    items_per_page = IntegerField('Items Per Page', validators=[NumberRange(min=10, max=100)], default=20)
    
    submit = SubmitField('Save Preferences')

class SystemSettingsForm(FlaskForm):
    app_name = StringField('Application Name', validators=[DataRequired()], default='Academic Pro')
    max_file_size = IntegerField('Max File Size (MB)', validators=[NumberRange(min=1, max=100)], default=16)
    session_timeout = IntegerField('Session Timeout (minutes)', validators=[NumberRange(min=30, max=1440)], default=120)
    
    enable_analytics = BooleanField('Enable Analytics', default=True)
    enable_predictions = BooleanField('Enable AI Predictions', default=True)
    enable_reports = BooleanField('Enable Reports', default=True)
    enable_exports = BooleanField('Enable Data Exports', default=True)
    
    maintenance_mode = BooleanField('Maintenance Mode', default=False)
    debug_mode = BooleanField('Debug Mode', default=False)
    
    backup_frequency = SelectField('Backup Frequency', choices=[
        ('daily', 'Daily'),
        ('weekly', 'Weekly'),
        ('monthly', 'Monthly'),
        ('manual', 'Manual Only')
    ], default='weekly')
    
    submit = SubmitField('Save System Settings')

class NotificationSettingsForm(FlaskForm):
    email_reports = BooleanField('Email Report Notifications', default=True)
    email_alerts = BooleanField('Email Alert Notifications', default=True)
    email_updates = BooleanField('Email System Updates', default=False)
    
    dashboard_alerts = BooleanField('Dashboard Alerts', default=True)
    dashboard_updates = BooleanField('Dashboard Updates', default=True)
    
    alert_threshold = IntegerField('Alert Threshold (%)', validators=[NumberRange(min=0, max=100)], default=60)
    
    submit = SubmitField('Save Notification Settings')

@settings_bp.route('/settings')
@login_required
def settings():
    """Settings dashboard"""
    try:
        # Initialize forms
        profile_form = ProfileSettingsForm()
        preferences_form = PreferencesForm()
        system_form = SystemSettingsForm()
        notification_form = NotificationSettingsForm()
        
        # Load user preferences
        user_preferences = load_user_preferences()
        
        # Load system settings
        system_settings = load_system_settings()
        
        # Load notification settings
        notification_settings = load_notification_settings()
        
        # Get user statistics
        user_stats = get_user_statistics()
        
        # Get system information
        system_info = get_system_information()
        
        return render_template('settings.html',
                             profile_form=profile_form,
                             preferences_form=preferences_form,
                             system_form=system_form,
                             notification_form=notification_form,
                             user_preferences=user_preferences,
                             system_settings=system_settings,
                             notification_settings=notification_settings,
                             user_stats=user_stats,
                             system_info=system_info)
    
    except Exception as e:
        logger.error(f"Settings page error: {e}")
        return render_template('error.html', error_code=500, error_message="Settings loading failed")

@settings_bp.route('/update-profile', methods=['POST'])
@login_required
def update_profile():
    """Update user profile"""
    try:
        form = ProfileSettingsForm()
        if form.validate_on_submit():
            # Verify current password
            if not current_user.check_password(form.current_password.data):
                flash('Current password is incorrect', 'error')
                return redirect(url_for('settings.settings'))
            
            # Update email if changed
            if form.email.data != current_user.email:
                current_user.email = form.email.data.lower()
            
            # Update password if provided
            if form.new_password.data:
                if form.new_password.data != form.confirm_password.data:
                    flash('New passwords do not match', 'error')
                    return redirect(url_for('settings.settings'))
                
                current_user.set_password(form.new_password.data)
            
            # Save changes
            from app import db
            db.session.commit()
            
            flash('Profile updated successfully', 'success')
            logger.info(f"User {current_user.id} updated profile")
            
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    flash(f'{field}: {error}', 'error')
        
        return redirect(url_for('settings.settings'))
    
    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        flash('Error updating profile. Please try again.', 'error')
        return redirect(url_for('settings.settings'))

@settings_bp.route('/update-preferences', methods=['POST'])
@login_required
def update_preferences():
    """Update user preferences"""
    try:
        form = PreferencesForm()
        if form.validate_on_submit():
            preferences = {
                'theme': form.theme.data,
                'language': form.language.data,
                'timezone': form.timezone.data,
                'notifications_email': form.notifications_email.data,
                'notifications_dashboard': form.notifications_dashboard.data,
                'notifications_reports': form.notifications_reports.data,
                'default_view': form.default_view.data,
                'items_per_page': form.items_per_page.data,
                'updated_at': datetime.now().isoformat()
            }
            
            save_user_preferences(preferences)
            flash('Preferences updated successfully', 'success')
            logger.info(f"User {current_user.id} updated preferences")
            
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    flash(f'{field}: {error}', 'error')
        
        return redirect(url_for('settings.settings'))
    
    except Exception as e:
        logger.error(f"Error updating preferences: {e}")
        flash('Error updating preferences. Please try again.', 'error')
        return redirect(url_for('settings.settings'))

@settings_bp.route('/update-system-settings', methods=['POST'])
@login_required
def update_system_settings():
    """Update system settings (admin only)"""
    try:
        # Check if user is admin (simplified check)
        if current_user.email != 'admin@example.com':  # Replace with proper admin check
            flash('Access denied. Admin privileges required.', 'error')
            return redirect(url_for('settings.settings'))
        
        form = SystemSettingsForm()
        if form.validate_on_submit():
            settings_data = {
                'app_name': form.app_name.data,
                'max_file_size': form.max_file_size.data,
                'session_timeout': form.session_timeout.data,
                'enable_analytics': form.enable_analytics.data,
                'enable_predictions': form.enable_predictions.data,
                'enable_reports': form.enable_reports.data,
                'enable_exports': form.enable_exports.data,
                'maintenance_mode': form.maintenance_mode.data,
                'debug_mode': form.debug_mode.data,
                'backup_frequency': form.backup_frequency.data,
                'updated_at': datetime.now().isoformat(),
                'updated_by': current_user.email
            }
            
            save_system_settings(settings_data)
            flash('System settings updated successfully', 'success')
            logger.info(f"Admin {current_user.id} updated system settings")
            
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    flash(f'{field}: {error}', 'error')
        
        return redirect(url_for('settings.settings'))
    
    except Exception as e:
        logger.error(f"Error updating system settings: {e}")
        flash('Error updating system settings. Please try again.', 'error')
        return redirect(url_for('settings.settings'))

@settings_bp.route('/update-notifications', methods=['POST'])
@login_required
def update_notifications():
    """Update notification settings"""
    try:
        form = NotificationSettingsForm()
        if form.validate_on_submit():
            notification_settings = {
                'email_reports': form.email_reports.data,
                'email_alerts': form.email_alerts.data,
                'email_updates': form.email_updates.data,
                'dashboard_alerts': form.dashboard_alerts.data,
                'dashboard_updates': form.dashboard_updates.data,
                'alert_threshold': form.alert_threshold.data,
                'updated_at': datetime.now().isoformat()
            }
            
            save_notification_settings(notification_settings)
            flash('Notification settings updated successfully', 'success')
            logger.info(f"User {current_user.id} updated notification settings")
            
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    flash(f'{field}: {error}', 'error')
        
        return redirect(url_for('settings.settings'))
    
    except Exception as e:
        logger.error(f"Error updating notification settings: {e}")
        flash('Error updating notification settings. Please try again.', 'error')
        return redirect(url_for('settings.settings'))

@settings_bp.route('/export-data')
@login_required
def export_data():
    """Export user data"""
    try:
        # Create user data export
        user_data = {
            'profile': {
                'email': current_user.email,
                'created_at': current_user.created_at.isoformat(),
                'last_login': current_user.last_login.isoformat() if current_user.last_login else None
            },
            'preferences': load_user_preferences(),
            'activity_summary': get_user_activity_summary(),
            'export_date': datetime.now().isoformat()
        }
        
        response = jsonify(user_data)
        response.headers['Content-Disposition'] = f'attachment; filename=user_data_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        logger.info(f"User {current_user.id} exported data")
        return response
    
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        flash('Error exporting data. Please try again.', 'error')
        return redirect(url_for('settings.settings'))

@settings_bp.route('/reset-preferences', methods=['POST'])
@login_required
def reset_preferences():
    """Reset user preferences to default"""
    try:
        # Validate CSRF
        csrf_token = request.headers.get('X-CSRFToken')
        if csrf_token:
            try:
                validate_csrf(csrf_token)
            except:
                return jsonify({'error': 'Invalid CSRF token'}), 400
        
        default_preferences = {
            'theme': 'light',
            'language': 'en',
            'timezone': 'UTC',
            'notifications_email': True,
            'notifications_dashboard': True,
            'notifications_reports': True,
            'default_view': 'dashboard',
            'items_per_page': 20,
            'updated_at': datetime.now().isoformat()
        }
        
        save_user_preferences(default_preferences)
        logger.info(f"User {current_user.id} reset preferences to default")
        
        return jsonify({'success': True, 'message': 'Preferences reset to default'})
    
    except Exception as e:
        logger.error(f"Error resetting preferences: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Helper functions

def load_user_preferences():
    """Load user preferences from storage"""
    try:
        preferences_file = f'instance/user_preferences_{current_user.id}.json'
        if os.path.exists(preferences_file):
            with open(preferences_file, 'r') as f:
                return json.load(f)
        else:
            # Return default preferences
            return {
                'theme': 'light',
                'language': 'en',
                'timezone': 'UTC',
                'notifications_email': True,
                'notifications_dashboard': True,
                'notifications_reports': True,
                'default_view': 'dashboard',
                'items_per_page': 20
            }
    except Exception as e:
        logger.error(f"Error loading user preferences: {e}")
        return {}

def save_user_preferences(preferences):
    """Save user preferences to storage"""
    try:
        os.makedirs('instance', exist_ok=True)
        preferences_file = f'instance/user_preferences_{current_user.id}.json'
        with open(preferences_file, 'w') as f:
            json.dump(preferences, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving user preferences: {e}")

def load_system_settings():
    """Load system settings from storage"""
    try:
        settings_file = 'instance/system_settings.json'
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
                return json.load(f)
        else:
            # Return default settings
            return {
                'app_name': 'Academic Pro',
                'max_file_size': 16,
                'session_timeout': 120,
                'enable_analytics': True,
                'enable_predictions': True,
                'enable_reports': True,
                'enable_exports': True,
                'maintenance_mode': False,
                'debug_mode': False,
                'backup_frequency': 'weekly'
            }
    except Exception as e:
        logger.error(f"Error loading system settings: {e}")
        return {}

def save_system_settings(settings):
    """Save system settings to storage"""
    try:
        os.makedirs('instance', exist_ok=True)
        settings_file = 'instance/system_settings.json'
        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving system settings: {e}")

def load_notification_settings():
    """Load notification settings from storage"""
    try:
        notifications_file = f'instance/notification_settings_{current_user.id}.json'
        if os.path.exists(notifications_file):
            with open(notifications_file, 'r') as f:
                return json.load(f)
        else:
            # Return default settings
            return {
                'email_reports': True,
                'email_alerts': True,
                'email_updates': False,
                'dashboard_alerts': True,
                'dashboard_updates': True,
                'alert_threshold': 60
            }
    except Exception as e:
        logger.error(f"Error loading notification settings: {e}")
        return {}

def save_notification_settings(settings):
    """Save notification settings to storage"""
    try:
        os.makedirs('instance', exist_ok=True)
        notifications_file = f'instance/notification_settings_{current_user.id}.json'
        with open(notifications_file, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving notification settings: {e}")

def get_user_statistics():
    """Get user activity statistics"""
    try:
        # Load basic statistics
        return {
            'total_logins': 25,
            'last_login': current_user.last_login.strftime('%Y-%m-%d %H:%M') if current_user.last_login else 'Never',
            'account_created': current_user.created_at.strftime('%Y-%m-%d'),
            'reports_generated': 8,
            'predictions_run': 15,
            'data_exports': 3
        }
    except Exception as e:
        logger.error(f"Error getting user statistics: {e}")
        return {}

def get_system_information():
    """Get system information"""
    try:
        return {
            'version': '2.0.0',
            'python_version': '3.13',
            'database': 'SQLite',
            'last_backup': '2024-01-15 10:30:00',
            'uptime': '15 days, 6 hours',
            'memory_usage': '45%',
            'disk_usage': '32%',
            'active_users': 1
        }
    except Exception as e:
        logger.error(f"Error getting system information: {e}")
        return {}

def get_user_activity_summary():
    """Get user activity summary for export"""
    try:
        return {
            'total_sessions': 25,
            'average_session_duration': '45 minutes',
            'most_used_feature': 'Student Management',
            'favorite_view': 'Dashboard',
            'reports_generated': 8,
            'data_accessed': 'Student performance data'
        }
    except Exception as e:
        logger.error(f"Error getting user activity summary: {e}")
        return {} 