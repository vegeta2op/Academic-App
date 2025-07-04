from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify
from flask_login import login_required, current_user
from flask_wtf import FlaskForm
from flask_wtf.csrf import validate_csrf
from wtforms import StringField, SubmitField, IntegerField, SelectField, TextAreaField
from wtforms.validators import DataRequired, NumberRange, Length, Email
import pandas as pd
import json
import os
import logging
from datetime import datetime
import numpy as np

# Import our advanced prediction service
from services.prediction_service import prediction_service

student_bp = Blueprint('student', __name__)
logger = logging.getLogger(__name__)

class StudentSearchForm(FlaskForm):
    search_query = StringField('Search Students', validators=[Length(max=100)])
    search_type = SelectField('Search By', choices=[
        ('name', 'Name'),
        ('usn', 'USN'),
        ('performance', 'Performance Level')
    ], default='name')
    submit = SubmitField('Search')

class AddStudentForm(FlaskForm):
    name = StringField('Student Name', validators=[DataRequired(), Length(min=2, max=100)])
    usn = StringField('USN', validators=[DataRequired(), Length(min=3, max=20)])
    sem1 = IntegerField('Semester 1', validators=[DataRequired(), NumberRange(min=0, max=100)])
    sem2 = IntegerField('Semester 2', validators=[DataRequired(), NumberRange(min=0, max=100)])
    sem3 = IntegerField('Semester 3', validators=[DataRequired(), NumberRange(min=0, max=100)])
    sem4 = IntegerField('Semester 4', validators=[DataRequired(), NumberRange(min=0, max=100)])
    sem5 = IntegerField('Semester 5', validators=[DataRequired(), NumberRange(min=0, max=100)])
    sem6 = IntegerField('Semester 6', validators=[DataRequired(), NumberRange(min=0, max=100)])
    submit = SubmitField('Add Student')

class StudentNotesForm(FlaskForm):
    notes = TextAreaField('Notes', validators=[Length(max=1000)])
    submit = SubmitField('Save Notes')

def get_student_statistics():
    """Get comprehensive statistics about students"""
    try:
        df = pd.read_csv('data.csv')
        
        stats = {
            'total_students': len(df),
            'average_grades': {
                'sem1': round(df['sem1'].mean(), 2),
                'sem2': round(df['sem2'].mean(), 2),
                'sem3': round(df['sem3'].mean(), 2),
                'sem4': round(df['sem4'].mean(), 2),
                'sem5': round(df['sem5'].mean(), 2),
                'sem6': round(df['sem6'].mean(), 2)
            },
            'performance_distribution': {
                'excellent': len(df[df['sem6'] >= 90]),
                'good': len(df[(df['sem6'] >= 80) & (df['sem6'] < 90)]),
                'average': len(df[(df['sem6'] >= 70) & (df['sem6'] < 80)]),
                'below_average': len(df[df['sem6'] < 70])
            },
            'top_performers': df.nlargest(10, 'sem6')[['Name', 'USN', 'sem6']].to_dict('records'),
            'improvement_trends': []
        }
        
        # Calculate improvement trends
        for _, student in df.iterrows():
            improvement = student['sem6'] - student['sem1']
            stats['improvement_trends'].append({
                'name': student['Name'],
                'usn': student['USN'],
                'improvement': improvement,
                'category': 'improved' if improvement > 0 else 'declined' if improvement < 0 else 'stable'
            })
        
        return stats
    
    except Exception as e:
        logger.error(f"Error getting student statistics: {e}")
        return None

def search_students(query, search_type):
    """Search students based on query and type"""
    try:
        df = pd.read_csv('data.csv')
        
        if not query:
            return df.to_dict('records')
        
        if search_type == 'name':
            filtered_df = df[df['Name'].str.contains(query, case=False, na=False)]
        elif search_type == 'usn':
            filtered_df = df[df['USN'].astype(str).str.contains(query, case=False, na=False)]
        elif search_type == 'performance':
            # Search by performance level
            if query.lower() in ['excellent', 'high']:
                filtered_df = df[df['sem6'] >= 90]
            elif query.lower() in ['good', 'medium']:
                filtered_df = df[(df['sem6'] >= 80) & (df['sem6'] < 90)]
            elif query.lower() in ['average', 'normal']:
                filtered_df = df[(df['sem6'] >= 70) & (df['sem6'] < 80)]
            elif query.lower() in ['below', 'low', 'poor']:
                filtered_df = df[df['sem6'] < 70]
            else:
                filtered_df = df
        else:
            filtered_df = df
        
        return filtered_df.to_dict('records')
    
    except Exception as e:
        logger.error(f"Error searching students: {e}")
        return []

def load_student_notes(usn):
    """Load notes for a specific student"""
    try:
        notes_file = f'student_notes_{usn}.json'
        if os.path.exists(notes_file):
            with open(notes_file, 'r') as f:
                return json.load(f)
        return {"notes": ""}
    except Exception as e:
        logger.error(f"Error loading student notes: {e}")
        return {"notes": ""}

def save_student_notes(usn, notes):
    """Save notes for a specific student"""
    try:
        notes_file = f'student_notes_{usn}.json'
        with open(notes_file, 'w') as f:
            json.dump({"notes": notes, "updated_by": current_user.email, "updated_at": datetime.now().isoformat()}, f)
        return True
    except Exception as e:
        logger.error(f"Error saving student notes: {e}")
        return False

@student_bp.route("/students")
@login_required
def students():
    """Enhanced students page with search and analytics"""
    try:
        # Get form data
        search_form = StudentSearchForm()
        add_form = AddStudentForm()
        
        # Get search parameters
        search_query = request.args.get('search_query', '')
        search_type = request.args.get('search_type', 'name')
        
        # Search students
        students_list = search_students(search_query, search_type)
        
        # Get statistics
        stats = get_student_statistics()
        
        # Get recent activities
        recent_activities = [
            {"action": "New student added", "details": "John Doe (USN: 142)", "time": "2 hours ago"},
            {"action": "Student grades updated", "details": "Jane Smith", "time": "1 day ago"},
            {"action": "Student notes updated", "details": "Bob Johnson", "time": "2 days ago"}
        ]
        
        return render_template('students.html', 
                             students=students_list,
                             search_form=search_form,
                             add_form=add_form,
                             stats=stats,
                             recent_activities=recent_activities,
                             search_query=search_query,
                             search_type=search_type)
    
    except Exception as e:
        logger.error(f"Students page error: {e}")
        flash('Error loading students page. Please try again.', 'error')
        return render_template('error.html', error_code=500, error_message="Students page loading failed")

@student_bp.route("/add-student", methods=['POST'])
@login_required
def add_student():
    """Add a new student with validation"""
    form = AddStudentForm()
    
    if form.validate_on_submit():
        try:
            # Check if USN already exists
            df = pd.read_csv('data.csv')
            if form.usn.data in df['USN'].astype(str).values:
                flash('USN already exists. Please use a different USN.', 'error')
                return redirect(url_for('student.students'))
            
            # Create new student record
            new_student = {
                'Name': form.name.data,
                'USN': form.usn.data,
                'sem1': form.sem1.data,
                'sem2': form.sem2.data,
                'sem3': form.sem3.data,
                'sem4': form.sem4.data,
                'sem5': form.sem5.data,
                'sem6': form.sem6.data
            }
            
            # Add to dataframe
            new_df = pd.DataFrame([new_student])
            df = pd.concat([df, new_df], ignore_index=True)
            
            # Save to CSV
            df.to_csv('data.csv', index=False)
            
            # Log the action
            logger.info(f"User {current_user.id} added new student: {form.name.data} (USN: {form.usn.data})")
            
            flash('Student added successfully!', 'success')
            
            # Trigger model retraining if prediction service is available
            if hasattr(prediction_service, 'retrain_on_data_update'):
                prediction_service.retrain_on_data_update()
        
        except Exception as e:
            logger.error(f"Error adding student: {e}")
            flash('Error adding student. Please try again.', 'error')
    
    else:
        # Display form validation errors
        for field, errors in form.errors.items():
            for error in errors:
                flash(f'{field}: {error}', 'error')
    
    return redirect(url_for('student.students'))

@student_bp.route("/api/students/search")
@login_required
def api_search_students():
    """API endpoint for searching students"""
    try:
        query = request.args.get('q', '')
        search_type = request.args.get('type', 'name')
        
        students_list = search_students(query, search_type)
        
        return jsonify({
            'success': True,
            'students': students_list,
            'total': len(students_list)
        })
    
    except Exception as e:
        logger.error(f"Error in student search API: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@student_bp.route("/api/student/<usn>")
@login_required
def get_student_details(usn):
    """Get detailed information about a specific student"""
    try:
        df = pd.read_csv('data.csv')
        student = df[df['USN'].astype(str) == str(usn)]
        
        if student.empty:
            return jsonify({'error': 'Student not found'}), 404
        
        student_data = student.iloc[0].to_dict()
        
        # Get student notes
        notes = load_student_notes(usn)
        
        # Calculate additional metrics
        grades = [student_data['sem1'], student_data['sem2'], student_data['sem3'], 
                 student_data['sem4'], student_data['sem5'], student_data['sem6']]
        
        analytics = {
            'average_grade': round(np.mean(grades), 2),
            'highest_grade': max(grades),
            'lowest_grade': min(grades),
            'improvement': student_data['sem6'] - student_data['sem1'],
            'consistency': round(np.std(grades), 2)
        }
        
        # Get class rank
        df_sorted = df.sort_values('sem6', ascending=False)
        rank = df_sorted.index[df_sorted['USN'].astype(str) == str(usn)].tolist()[0] + 1
        
        result = {
            'success': True,
            'student': student_data,
            'notes': notes,
            'analytics': analytics,
            'rank': rank,
            'total_students': len(df)
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error getting student details: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@student_bp.route("/api/student/<usn>/notes", methods=['POST'])
@login_required
def save_student_notes_api(usn):
    """Save notes for a specific student"""
    try:
        # Validate CSRF
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
        
        notes = data.get('notes', '')
        
        # Validate input
        if len(notes) > 1000:
            return jsonify({'error': 'Notes too long (max 1000 characters)'}), 400
        
        # Check if student exists
        df = pd.read_csv('data.csv')
        if usn not in df['USN'].astype(str).values:
            return jsonify({'error': 'Student not found'}), 404
        
        # Save notes
        if save_student_notes(usn, notes):
            logger.info(f"User {current_user.id} saved notes for student USN: {usn}")
            return jsonify({'success': True, 'message': 'Notes saved successfully'})
        else:
            return jsonify({'error': 'Failed to save notes'}), 500
    
    except Exception as e:
        logger.error(f"Error saving student notes: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@student_bp.route("/api/student/<usn>/delete", methods=['POST'])
@login_required
def delete_student(usn):
    """Delete a student record"""
    try:
        # Validate CSRF
        csrf_token = request.headers.get('X-CSRFToken')
        if not csrf_token:
            return jsonify({'error': 'CSRF token missing'}), 400
        
        try:
            validate_csrf(csrf_token)
        except:
            return jsonify({'error': 'Invalid CSRF token'}), 400
        
        # Load data
        df = pd.read_csv('data.csv')
        
        # Check if student exists
        if usn not in df['USN'].astype(str).values:
            return jsonify({'error': 'Student not found'}), 404
        
        # Remove student
        df = df[df['USN'].astype(str) != str(usn)]
        
        # Save updated data
        df.to_csv('data.csv', index=False)
        
        # Delete student notes file if exists
        notes_file = f'student_notes_{usn}.json'
        if os.path.exists(notes_file):
            os.remove(notes_file)
        
        # Log the action
        logger.info(f"User {current_user.id} deleted student USN: {usn}")
        
        # Trigger model retraining if prediction service is available
        if hasattr(prediction_service, 'retrain_on_data_update'):
            prediction_service.retrain_on_data_update()
        
        return jsonify({'success': True, 'message': 'Student deleted successfully'})
    
    except Exception as e:
        logger.error(f"Error deleting student: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@student_bp.route("/api/students/stats")
@login_required
def get_student_stats():
    """Get real-time student statistics"""
    try:
        stats = get_student_statistics()
        if stats:
            return jsonify({
                'success': True,
                'stats': stats
            })
        else:
            return jsonify({'error': 'Failed to load statistics'}), 500
    
    except Exception as e:
        logger.error(f"Error getting student stats: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@student_bp.route("/api/students/performance-analysis")
@login_required
def get_performance_analysis():
    """Get detailed performance analysis of all students"""
    try:
        df = pd.read_csv('data.csv')
        
        analysis = {
            'semester_trends': {},
            'performance_categories': {},
            'improvement_analysis': {},
            'outliers': []
        }
        
        # Semester trends
        for sem in ['sem1', 'sem2', 'sem3', 'sem4', 'sem5', 'sem6']:
            analysis['semester_trends'][sem] = {
                'average': round(df[sem].mean(), 2),
                'median': round(df[sem].median(), 2),
                'std': round(df[sem].std(), 2),
                'min': int(df[sem].min()),
                'max': int(df[sem].max())
            }
        
        # Performance categories
        for grade_range, category in [(90, 'excellent'), (80, 'good'), (70, 'average'), (0, 'below_average')]:
            if category == 'below_average':
                count = len(df[df['sem6'] < 70])
            else:
                next_range = 100 if grade_range == 90 else grade_range + 10
                count = len(df[(df['sem6'] >= grade_range) & (df['sem6'] < next_range)])
            
            analysis['performance_categories'][category] = {
                'count': count,
                'percentage': round((count / len(df)) * 100, 2)
            }
        
        # Improvement analysis
        improvements = df['sem6'] - df['sem1']
        analysis['improvement_analysis'] = {
            'average_improvement': round(improvements.mean(), 2),
            'students_improved': len(improvements[improvements > 0]),
            'students_declined': len(improvements[improvements < 0]),
            'students_stable': len(improvements[improvements == 0])
        }
        
        # Outliers (students with unusual performance patterns)
        for _, student in df.iterrows():
            grades = [student['sem1'], student['sem2'], student['sem3'], 
                     student['sem4'], student['sem5'], student['sem6']]
            grade_std = np.std(grades)
            
            if grade_std > 15:  # High variability
                analysis['outliers'].append({
                    'name': student['Name'],
                    'usn': student['USN'],
                    'type': 'high_variability',
                    'std': round(grade_std, 2)
                })
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
    
    except Exception as e:
        logger.error(f"Error getting performance analysis: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@student_bp.route("/export/students-detailed")
@login_required
def export_students_detailed():
    """Export detailed student data with analytics"""
    try:
        df = pd.read_csv('data.csv')
        
        # Add calculated fields
        df['average_grade'] = df[['sem1', 'sem2', 'sem3', 'sem4', 'sem5', 'sem6']].mean(axis=1)
        df['improvement'] = df['sem6'] - df['sem1']
        df['consistency'] = df[['sem1', 'sem2', 'sem3', 'sem4', 'sem5', 'sem6']].std(axis=1)
        
        # Add performance category
        df['performance_category'] = pd.cut(df['sem6'], 
                                           bins=[0, 70, 80, 90, 100], 
                                           labels=['Below Average', 'Average', 'Good', 'Excellent'])
        
        # Add export metadata
        df['export_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df['exported_by'] = current_user.email
        
        # Convert to CSV
        csv_data = df.to_csv(index=False)
        
        logger.info(f"User {current_user.id} exported detailed student data")
        
        return jsonify({
            'success': True,
            'csv_data': csv_data,
            'filename': f'students_detailed_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        })
    
    except Exception as e:
        logger.error(f"Error exporting detailed student data: {e}")
        return jsonify({'error': 'Internal server error'}), 500
