from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify
from flask_login import login_required, current_user
from flask_wtf import FlaskForm
from flask_wtf.csrf import validate_csrf
from wtforms import StringField, SubmitField, IntegerField, TextAreaField
from wtforms.validators import DataRequired, NumberRange, Length
import pandas as pd
import csv
import json
import os
from datetime import datetime
import logging

# Import our advanced prediction service
from services.prediction_service import prediction_service

dashboard_bp = Blueprint('dashboard', __name__)
logger = logging.getLogger(__name__)

class UpdateMarksForm(FlaskForm):
    usn = StringField('USN', validators=[DataRequired(), Length(min=1, max=10)])
    sem1 = IntegerField('Semester 1', validators=[DataRequired(), NumberRange(min=0, max=100)])
    sem2 = IntegerField('Semester 2', validators=[DataRequired(), NumberRange(min=0, max=100)])
    sem3 = IntegerField('Semester 3', validators=[DataRequired(), NumberRange(min=0, max=100)])
    sem4 = IntegerField('Semester 4', validators=[DataRequired(), NumberRange(min=0, max=100)])
    sem5 = IntegerField('Semester 5', validators=[DataRequired(), NumberRange(min=0, max=100)])
    submit = SubmitField('Update Marks')

class NoteForm(FlaskForm):
    note_content = TextAreaField('Note Content', validators=[Length(max=500)])
    submit = SubmitField('Save Note')

def get_student_analytics():
    """Get comprehensive analytics for dashboard"""
    try:
        df = pd.read_csv('data.csv')
        
        analytics = {
            'total_students': len(df),
            'average_performance': df[['sem1', 'sem2', 'sem3', 'sem4', 'sem5', 'sem6']].mean().mean(),
            'top_performers': df.nlargest(5, 'sem6')[['Name', 'USN', 'sem6']].to_dict('records'),
            'improvement_trends': [],
            'performance_distribution': {},
            'semester_averages': df[['sem1', 'sem2', 'sem3', 'sem4', 'sem5', 'sem6']].mean().to_dict()
        }
        
        # Calculate improvement trends
        for _, student in df.iterrows():
            trend = student['sem6'] - student['sem1']
            analytics['improvement_trends'].append({
                'name': student['Name'],
                'usn': student['USN'],
                'trend': trend
            })
        
        # Performance distribution
        for grade_range, label in [(90, 'Excellent'), (80, 'Good'), (70, 'Average'), (60, 'Below Average')]:
            count = len(df[df['sem6'] >= grade_range])
            analytics['performance_distribution'][label] = count
        
        return analytics
    
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        return None

def load_user_notes():
    """Load user-specific notes"""
    try:
        notes_file = f'user_notes_{current_user.id}.json'
        if os.path.exists(notes_file):
            with open(notes_file, 'r') as f:
                return json.load(f)
        return {"note1": "To Do List", "note2": "To Do List", "note3": "To Do List", "note4": "To Do List"}
    except Exception as e:
        logger.error(f"Error loading notes: {e}")
        return {"note1": "To Do List", "note2": "To Do List", "note3": "To Do List", "note4": "To Do List"}

def save_user_notes(notes):
    """Save user-specific notes"""
    try:
        notes_file = f'user_notes_{current_user.id}.json'
        with open(notes_file, 'w') as f:
            json.dump(notes, f)
        return True
    except Exception as e:
        logger.error(f"Error saving notes: {e}")
        return False

@dashboard_bp.route("/dashboard")
@login_required
def dashboard():
    """Enhanced dashboard with analytics and modern UI"""
    try:
        # Load student data
        df = pd.read_csv('data.csv')
        students = df.to_dict('records')
        
        # Get analytics
        analytics = get_student_analytics()
        
        # Load user notes
        notes = load_user_notes()
        
        # Create forms
        form = UpdateMarksForm()
        note_form = NoteForm()
        
        # Get recent activities (this would be from audit logs in production)
        recent_activities = [
            {"action": "Student marks updated", "time": "2 hours ago"},
            {"action": "New prediction generated", "time": "1 day ago"},
            {"action": "Dashboard accessed", "time": "Just now"}
        ]
        
        return render_template('dashboard.html', 
                             students=students, 
                             form=form,
                             note_form=note_form,
                             analytics=analytics,
                             notes=notes,
                             recent_activities=recent_activities)
    
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        flash('Error loading dashboard. Please try again.', 'error')
        return render_template('error.html', error_code=500, error_message="Dashboard loading failed")

@dashboard_bp.route('/update-marks', methods=['POST'])
@login_required
def update_marks():
    """Update student marks with validation and security"""
    form = UpdateMarksForm()
    
    if form.validate_on_submit():
        try:
            usn = form.usn.data
            new_marks = {
                'sem1': form.sem1.data,
                'sem2': form.sem2.data,
                'sem3': form.sem3.data,
                'sem4': form.sem4.data,
                'sem5': form.sem5.data
            }
            
            # Validate USN exists
            df = pd.read_csv('data.csv')
            if usn not in df['USN'].astype(str).values:
                flash('Student USN not found', 'error')
                return redirect(url_for('dashboard.dashboard'))
            
            # Update the CSV file safely
            students = df.to_dict('records')
            student_found = False
            
            for student in students:
                if str(student['USN']) == str(usn):
                    student.update(new_marks)
                    student_found = True
                    break
            
            if student_found:
                # Write back to CSV
                updated_df = pd.DataFrame(students)
                updated_df.to_csv('data.csv', index=False)
                
                # Log the action
                logger.info(f"User {current_user.id} updated marks for USN {usn}")
                
                flash('Marks updated successfully!', 'success')
                
                # Trigger model retraining if needed
                if hasattr(prediction_service, 'retrain_on_data_update'):
                    prediction_service.retrain_on_data_update()
            
            else:
                flash('Student not found', 'error')
        
        except Exception as e:
            logger.error(f"Error updating marks: {e}")
            flash('Error updating marks. Please try again.', 'error')
    
    else:
        # Display form validation errors
        for field, errors in form.errors.items():
            for error in errors:
                flash(f'{field}: {error}', 'error')
    
    return redirect(url_for('dashboard.dashboard'))

@dashboard_bp.route('/api/save-note', methods=['POST'])
@login_required
def save_note():
    """Save user notes via API"""
    try:
        # Validate CSRF for AJAX requests
        csrf_token = request.headers.get('X-CSRFToken')
        if not csrf_token:
            return jsonify({'error': 'CSRF token missing'}), 400
        
        try:
            validate_csrf(csrf_token)
        except:
            return jsonify({'error': 'Invalid CSRF token'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        note_id = data.get('noteId')
        content = data.get('content', '')
        
        # Validate input
        if not note_id or not isinstance(note_id, (int, str)):
            return jsonify({'error': 'Invalid note ID'}), 400
        
        if len(content) > 500:
            return jsonify({'error': 'Note content too long'}), 400
        
        # Load existing notes
        notes = load_user_notes()
        note_key = f'note{note_id}'
        
        if note_key in notes:
            notes[note_key] = content
            if save_user_notes(notes):
                logger.info(f"User {current_user.id} saved note {note_id}")
                return jsonify({'success': True, 'message': 'Note saved successfully'})
            else:
                return jsonify({'error': 'Failed to save note'}), 500
        else:
            return jsonify({'error': 'Invalid note ID'}), 400
    
    except Exception as e:
        logger.error(f"Error saving note: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@dashboard_bp.route('/api/analytics')
@login_required
def get_analytics():
    """Get real-time analytics data"""
    try:
        analytics = get_student_analytics()
        if analytics:
            return jsonify(analytics)
        else:
            return jsonify({'error': 'Failed to load analytics'}), 500
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@dashboard_bp.route('/api/retrain-models', methods=['POST'])
@login_required
def retrain_models():
    """Retrain machine learning models with latest data"""
    try:
        # Validate CSRF
        csrf_token = request.headers.get('X-CSRFToken')
        if not csrf_token:
            return jsonify({'error': 'CSRF token missing'}), 400
        
        try:
            validate_csrf(csrf_token)
        except:
            return jsonify({'error': 'Invalid CSRF token'}), 400
        
        from services.prediction_service import prediction_service
        
        # Retrain models
        success = prediction_service.train_full_pipeline()
        
        if success:
            logger.info(f"User {current_user.id} initiated model retraining")
            return jsonify({'success': True, 'message': 'Models retrained successfully'})
        else:
            return jsonify({'success': False, 'error': 'Model retraining failed'}), 500
    
    except Exception as e:
        logger.error(f"Error retraining models: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@dashboard_bp.route('/api/system-health')
@login_required
def get_system_health():
    """Get system health and performance metrics"""
    try:
        try:
            import psutil
            import sys
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get application metrics
            df = pd.read_csv('data.csv')
            
            health_data = {
                'system': {
                    'cpu_usage': cpu_percent,
                    'memory_usage': memory.percent,
                    'memory_available': memory.available // (1024*1024),  # MB
                    'disk_usage': disk.percent,
                    'disk_free': disk.free // (1024*1024*1024),  # GB
                    'uptime': 'Online'
                },
                'application': {
                    'total_students': len(df),
                    'models_status': 'Active',
                    'last_update': datetime.now().isoformat(),
                    'python_version': sys.version.split()[0],
                    'data_integrity': 'Good' if len(df) > 0 else 'Warning'
                },
                'alerts': []
            }
            
            # Add alerts based on thresholds
            if cpu_percent > 80:
                health_data['alerts'].append({
                    'type': 'warning',
                    'message': f'High CPU usage: {cpu_percent}%'
                })
            
            if memory.percent > 85:
                health_data['alerts'].append({
                    'type': 'warning',
                    'message': f'High memory usage: {memory.percent}%'
                })
            
            if disk.percent > 90:
                health_data['alerts'].append({
                    'type': 'error',
                    'message': f'Low disk space: {disk.percent}% used'
                })
            
            return jsonify({
                'success': True,
                'health': health_data,
                'timestamp': datetime.now().isoformat()
            })
        
        except ImportError:
            # psutil not available
            return jsonify({
                'success': True,
                'health': {
                    'system': {'status': 'Monitoring unavailable'},
                    'application': {'status': 'Running'},
                    'alerts': []
                },
                'timestamp': datetime.now().isoformat()
            })
    
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@dashboard_bp.route('/export/dashboard-report')
@login_required
def export_dashboard_report():
    """Export comprehensive dashboard report"""
    try:
        df = pd.read_csv('data.csv')
        
        # Create comprehensive report
        report_data = {
            'summary': {
                'total_students': len(df),
                'average_performance': df['sem6'].mean(),
                'report_generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'generated_by': current_user.email
            },
            'performance_metrics': {
                'semester_averages': {
                    'sem1': df['sem1'].mean(),
                    'sem2': df['sem2'].mean(),
                    'sem3': df['sem3'].mean(),
                    'sem4': df['sem4'].mean(),
                    'sem5': df['sem5'].mean(),
                    'sem6': df['sem6'].mean()
                },
                'grade_distribution': {
                    'excellent': len(df[df['sem6'] >= 90]),
                    'good': len(df[(df['sem6'] >= 80) & (df['sem6'] < 90)]),
                    'average': len(df[(df['sem6'] >= 70) & (df['sem6'] < 80)]),
                    'below_average': len(df[df['sem6'] < 70])
                }
            },
            'insights': {
                'top_performers': df.nlargest(5, 'sem6')[['Name', 'USN', 'sem6']].to_dict('records'),
                'most_improved': df.nlargest(5, df['sem6'] - df['sem1'])[['Name', 'USN', 'sem1', 'sem6']].to_dict('records'),
                'needs_attention': df[df['sem6'] < 60][['Name', 'USN', 'sem6']].to_dict('records')
            }
        }
        
        # Convert to JSON format for download
        import json
        json_report = json.dumps(report_data, indent=2, ensure_ascii=False)
        
        # Log the export
        logger.info(f"User {current_user.id} exported dashboard report")
        
        # Return as downloadable file
        from flask import Response
        return Response(
            json_report,
            mimetype='application/json',
            headers={
                'Content-Disposition': f'attachment; filename=dashboard_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            }
        )
    
    except Exception as e:
        logger.error(f"Error exporting dashboard report: {e}")
        flash('Error generating report. Please try again.', 'error')
        return redirect(url_for('dashboard.dashboard'))

@dashboard_bp.route('/api/notifications')
@login_required
def get_notifications():
    """Get user notifications and system alerts"""
    try:
        df = pd.read_csv('data.csv')
        
        notifications = []
        
        # Check for students needing attention
        low_performers = df[df['sem6'] < 60]
        if len(low_performers) > 0:
            notifications.append({
                'type': 'warning',
                'title': 'Students Need Attention',
                'message': f'{len(low_performers)} students have grades below 60%',
                'timestamp': datetime.now().isoformat(),
                'action_url': '/students?filter=low_performance'
            })
        
        # Check for data freshness
        if len(df) == 0:
            notifications.append({
                'type': 'error',
                'title': 'No Student Data',
                'message': 'No student records found in the system',
                'timestamp': datetime.now().isoformat(),
                'action_url': '/students'
            })
        
        # Check for system updates
        notifications.append({
            'type': 'info',
            'title': 'System Update Available',
            'message': 'New features and improvements are available',
            'timestamp': datetime.now().isoformat(),
            'action_url': '#'
        })
        
        # Achievement notifications
        excellent_students = len(df[df['sem6'] >= 90])
        if excellent_students > 0:
            notifications.append({
                'type': 'success',
                'title': 'Excellent Performance',
                'message': f'{excellent_students} students achieved excellent grades!',
                'timestamp': datetime.now().isoformat(),
                'action_url': '/students?filter=excellent'
            })
        
        return jsonify({
            'success': True,
            'notifications': notifications,
            'unread_count': len([n for n in notifications if n['type'] in ['warning', 'error']])
        })
    
    except Exception as e:
        logger.error(f"Error getting notifications: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@dashboard_bp.route('/api/predictions/batch', methods=['POST'])
@login_required
def batch_predictions():
    """Generate predictions for multiple students"""
    try:
        # Validate CSRF
        csrf_token = request.headers.get('X-CSRFToken')
        if not csrf_token:
            return jsonify({'error': 'CSRF token missing'}), 400
        
        try:
            validate_csrf(csrf_token)
        except:
            return jsonify({'error': 'Invalid CSRF token'}), 400
        
        # Load student data
        df = pd.read_csv('data.csv')
        students = df.to_dict('records')
        
        # Initialize prediction service if not already trained
        if not prediction_service.is_trained:
            if os.path.exists('models'):
                prediction_service.load_models()
            else:
                prediction_service.train_full_pipeline()
        
        # Generate predictions
        predictions = prediction_service.batch_predict(students)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'total': len(predictions)
        })
    
    except Exception as e:
        logger.error(f"Error generating batch predictions: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@dashboard_bp.route('/api/student/<usn>/insights')
@login_required
def get_student_insights(usn):
    """Get AI-powered insights for a specific student"""
    try:
        df = pd.read_csv('data.csv')
        student = df[df['USN'] == int(usn)]
        
        if student.empty:
            return jsonify({'error': 'Student not found'}), 404
        
        student_data = student.iloc[0].to_dict()
        
        # Initialize prediction service if needed
        if not prediction_service.is_trained:
            if os.path.exists('models'):
                prediction_service.load_models()
            else:
                prediction_service.train_full_pipeline()
        
        # Get insights
        insights = prediction_service.generate_performance_insights(student_data)
        prediction_result = prediction_service.get_prediction_confidence(student_data)
        
        return jsonify({
            'success': True,
            'insights': insights,
            'prediction': prediction_result,
            'student': student_data
        })
    
    except Exception as e:
        logger.error(f"Error getting student insights: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@dashboard_bp.route('/api/performance-chart/<usn>')
@login_required
def get_performance_chart(usn):
    """Get performance chart data for a student"""
    try:
        df = pd.read_csv('data.csv')
        student = df[df['USN'] == int(usn)]
        
        if student.empty:
            return jsonify({'error': 'Student not found'}), 404
        
        student_data = student.iloc[0].to_dict()
        
        # Prepare chart data
        semesters = ['Sem 1', 'Sem 2', 'Sem 3', 'Sem 4', 'Sem 5', 'Sem 6']
        grades = [
            student_data['sem1'], student_data['sem2'], student_data['sem3'],
            student_data['sem4'], student_data['sem5'], student_data['sem6']
        ]
        
        return jsonify({
            'success': True,
            'chart_data': {
                'labels': semesters,
                'grades': grades,
                'name': student_data['Name'],
                'usn': student_data['USN']
            }
        })
    
    except Exception as e:
        logger.error(f"Error getting performance chart: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@dashboard_bp.route('/export/students')
@login_required
def export_students():
    """Export student data as CSV"""
    try:
        df = pd.read_csv('data.csv')
        
        # Create a more detailed export
        export_data = df.copy()
        export_data['export_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        export_data['exported_by'] = current_user.email
        
        # Convert to CSV
        csv_data = export_data.to_csv(index=False)
        
        logger.info(f"User {current_user.id} exported student data")
        
        return jsonify({
            'success': True,
            'csv_data': csv_data,
            'filename': f'students_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        })
    
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return jsonify({'error': 'Internal server error'}), 500