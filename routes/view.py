from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify
from flask_login import login_required, current_user
from flask_wtf import FlaskForm
from flask_wtf.csrf import validate_csrf
from wtforms import SelectField, SubmitField
from wtforms.validators import DataRequired
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Import our advanced prediction service
from services.prediction_service import prediction_service

view_bp = Blueprint('view', __name__)
logger = logging.getLogger(__name__)

class VisualizationForm(FlaskForm):
    chart_type = SelectField('Chart Type', choices=[
        ('performance_trend', 'Performance Trend'),
        ('prediction_analysis', 'Prediction Analysis'),
        ('comparison', 'Class Comparison'),
        ('distribution', 'Grade Distribution'),
        ('correlation', 'Correlation Matrix')
    ], default='performance_trend')
    
    model_type = SelectField('Prediction Model', choices=[
        ('ensemble', 'Ensemble Model'),
        ('random_forest', 'Random Forest'),
        ('gradient_boosting', 'Gradient Boosting'),
        ('ridge', 'Ridge Regression')
    ], default='ensemble')
    
    submit = SubmitField('Generate Visualization')

def initialize_prediction_service():
    """Initialize the prediction service if not already done"""
    try:
        if not prediction_service.is_trained:
            if os.path.exists('models') and os.listdir('models'):
                success = prediction_service.load_models()
                if not success:
                    logger.info("Loading models failed, training new models...")
                    prediction_service.train_full_pipeline()
            else:
                logger.info("No existing models found, training new models...")
                prediction_service.train_full_pipeline()
        return True
    except Exception as e:
        logger.error(f"Error initializing prediction service: {e}")
        return False

def create_performance_trend_chart(student_data):
    """Create an interactive performance trend chart"""
    try:
        semesters = ['Sem 1', 'Sem 2', 'Sem 3', 'Sem 4', 'Sem 5', 'Sem 6']
        grades = [
            student_data['sem1'], student_data['sem2'], student_data['sem3'],
            student_data['sem4'], student_data['sem5'], student_data['sem6']
        ]
        
        # Get prediction for next semester
        if initialize_prediction_service():
            pred_result = prediction_service.get_prediction_confidence(student_data)
            if pred_result:
                semesters.append('Sem 7 (Predicted)')
                grades.append(pred_result['prediction'])
        
        fig = go.Figure()
        
        # Add actual grades
        fig.add_trace(go.Scatter(
            x=semesters[:-1] if len(semesters) > 6 else semesters,
            y=grades[:-1] if len(grades) > 6 else grades,
            mode='lines+markers',
            name='Actual Grades',
            line=dict(color='#3498db', width=3),
            marker=dict(size=10, color='#3498db')
        ))
        
        # Add predicted grade if available
        if len(semesters) > 6:
            fig.add_trace(go.Scatter(
                x=[semesters[-2], semesters[-1]],
                y=[grades[-2], grades[-1]],
                mode='lines+markers',
                name='Predicted Grade',
                line=dict(color='#e74c3c', width=3, dash='dash'),
                marker=dict(size=10, color='#e74c3c')
            ))
        
        # Add trend line
        x_numeric = list(range(len(grades)))
        z = np.polyfit(x_numeric, grades, 1)
        trend_line = np.poly1d(z)
        
        fig.add_trace(go.Scatter(
            x=semesters,
            y=[trend_line(i) for i in x_numeric],
            mode='lines',
            name='Trend Line',
            line=dict(color='#2ecc71', width=2, dash='dot'),
            opacity=0.7
        ))
        
        fig.update_layout(
            title=f"Academic Performance Trend - {student_data['Name']} (USN: {student_data['USN']})",
            xaxis_title='Semester',
            yaxis_title='Grade',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    except Exception as e:
        logger.error(f"Error creating performance trend chart: {e}")
        return None

def create_class_comparison_chart(student_data):
    """Create a chart comparing student with class average"""
    try:
        df = pd.read_csv('data.csv')
        
        semesters = ['sem1', 'sem2', 'sem3', 'sem4', 'sem5', 'sem6']
        semester_labels = ['Sem 1', 'Sem 2', 'Sem 3', 'Sem 4', 'Sem 5', 'Sem 6']
        
        # Student grades
        student_grades = [student_data[sem] for sem in semesters]
        
        # Class averages
        class_averages = [df[sem].mean() for sem in semesters]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=semester_labels,
            y=student_grades,
            mode='lines+markers',
            name=f"{student_data['Name']}",
            line=dict(color='#3498db', width=3),
            marker=dict(size=10)
        ))
        
        fig.add_trace(go.Scatter(
            x=semester_labels,
            y=class_averages,
            mode='lines+markers',
            name='Class Average',
            line=dict(color='#95a5a6', width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f"Performance vs Class Average - {student_data['Name']}",
            xaxis_title='Semester',
            yaxis_title='Grade',
            template='plotly_white',
            height=500
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    except Exception as e:
        logger.error(f"Error creating class comparison chart: {e}")
        return None

def create_prediction_analysis_chart(student_data):
    """Create detailed prediction analysis visualization"""
    try:
        if not initialize_prediction_service():
            return None
        
        pred_result = prediction_service.get_prediction_confidence(student_data)
        if not pred_result:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Predictions', 'Confidence Analysis', 'Feature Importance', 'Performance Metrics'),
            specs=[[{"type": "bar"}, {"type": "indicator"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Model predictions comparison
        model_names = ['Random Forest', 'XGBoost', 'Gradient Boosting', 'Ensemble']
        predictions = pred_result['individual_predictions']
        
        fig.add_trace(go.Bar(
            x=model_names,
            y=predictions,
            name='Predictions',
            marker_color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        ), row=1, col=1)
        
        # Confidence indicator
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=pred_result['confidence'] * 100,
            title={'text': "Confidence %"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "#2ecc71"},
                   'steps': [{'range': [0, 50], 'color': "#e74c3c"},
                            {'range': [50, 80], 'color': "#f39c12"},
                            {'range': [80, 100], 'color': "#2ecc71"}]}
        ), row=1, col=2)
        
        # Feature importance (if available)
        if 'random_forest' in prediction_service.feature_importance:
            importance = prediction_service.feature_importance['random_forest']
            features = list(importance.keys())[:5]  # Top 5 features
            values = [importance[f] for f in features]
            
            fig.add_trace(go.Bar(
                x=values,
                y=features,
                orientation='h',
                name='Importance',
                marker_color='#9b59b6'
            ), row=2, col=1)
        
        # Performance metrics over semesters
        semesters = ['Sem 1', 'Sem 2', 'Sem 3', 'Sem 4', 'Sem 5', 'Sem 6']
        grades = [student_data['sem1'], student_data['sem2'], student_data['sem3'],
                 student_data['sem4'], student_data['sem5'], student_data['sem6']]
        
        fig.add_trace(go.Scatter(
            x=semesters,
            y=grades,
            mode='lines+markers',
            name='Actual',
            line=dict(color='#3498db')
        ), row=2, col=2)
        
        fig.update_layout(height=800, title_text="Comprehensive Prediction Analysis")
        
        return fig.to_html(include_plotlyjs='cdn')
    
    except Exception as e:
        logger.error(f"Error creating prediction analysis chart: {e}")
        return None

def create_grade_distribution_chart():
    """Create grade distribution visualization for the entire class"""
    try:
        df = pd.read_csv('data.csv')
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Sem 1', 'Sem 2', 'Sem 3', 'Sem 4', 'Sem 5', 'Sem 6'),
            specs=[[{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}]]
        )
        
        semesters = ['sem1', 'sem2', 'sem3', 'sem4', 'sem5', 'sem6']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        
        for i, (sem, color) in enumerate(zip(semesters, colors)):
            row = (i // 3) + 1
            col = (i % 3) + 1
            
            fig.add_trace(go.Histogram(
                x=df[sem],
                name=f'Sem {i+1}',
                marker_color=color,
                opacity=0.7,
                nbinsx=10
            ), row=row, col=col)
        
        fig.update_layout(
            height=600,
            title_text="Grade Distribution Across Semesters",
            showlegend=False
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    except Exception as e:
        logger.error(f"Error creating grade distribution chart: {e}")
        return None

def create_correlation_matrix():
    """Create correlation matrix visualization"""
    try:
        df = pd.read_csv('data.csv')
        
        # Select numeric columns
        numeric_cols = ['sem1', 'sem2', 'sem3', 'sem4', 'sem5', 'sem6']
        correlation_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Semester Grade Correlation Matrix",
            template='plotly_white',
            height=500
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    except Exception as e:
        logger.error(f"Error creating correlation matrix: {e}")
        return None

def get_student_insights(student_data):
    """Get comprehensive insights about the student"""
    try:
        insights = {
            'academic_insights': [],
            'prediction_insights': [],
            'improvement_suggestions': [],
            'risk_assessment': 'low'
        }
        
        # Academic insights
        grades = [student_data['sem1'], student_data['sem2'], student_data['sem3'],
                 student_data['sem4'], student_data['sem5'], student_data['sem6']]
        
        avg_grade = np.mean(grades)
        grade_trend = np.polyfit(range(len(grades)), grades, 1)[0]
        consistency = np.std(grades)
        
        if avg_grade >= 90:
            insights['academic_insights'].append("🌟 Excellent overall performance")
        elif avg_grade >= 80:
            insights['academic_insights'].append("👍 Good academic performance")
        elif avg_grade >= 70:
            insights['academic_insights'].append("📚 Average performance level")
        else:
            insights['academic_insights'].append("⚠️ Below average performance")
            insights['risk_assessment'] = 'high'
        
        if grade_trend > 2:
            insights['academic_insights'].append("📈 Strong upward trend")
        elif grade_trend > 0:
            insights['academic_insights'].append("📊 Positive improvement")
        elif grade_trend < -2:
            insights['academic_insights'].append("📉 Concerning downward trend")
            insights['risk_assessment'] = 'high'
        
        if consistency < 5:
            insights['academic_insights'].append("🎯 Very consistent performance")
        elif consistency > 10:
            insights['academic_insights'].append("⚡ Variable performance patterns")
        
        # Prediction insights
        if initialize_prediction_service():
            pred_result = prediction_service.get_prediction_confidence(student_data)
            if pred_result:
                confidence = pred_result['confidence']
                prediction = pred_result['prediction']
                
                if confidence > 0.8:
                    insights['prediction_insights'].append(f"🔮 High confidence prediction: {prediction:.1f}")
                elif confidence > 0.6:
                    insights['prediction_insights'].append(f"📊 Moderate confidence prediction: {prediction:.1f}")
                else:
                    insights['prediction_insights'].append(f"❓ Low confidence prediction: {prediction:.1f}")
                
                if prediction > avg_grade:
                    insights['prediction_insights'].append("📈 Expected to improve next semester")
                elif prediction < avg_grade - 5:
                    insights['prediction_insights'].append("⚠️ May face challenges next semester")
                    insights['risk_assessment'] = 'medium'
        
        # Improvement suggestions
        if avg_grade < 70:
            insights['improvement_suggestions'].append("📝 Consider additional tutoring support")
            insights['improvement_suggestions'].append("📚 Review study methods and time management")
        
        if consistency > 10:
            insights['improvement_suggestions'].append("🎯 Focus on consistent study habits")
        
        if grade_trend < 0:
            insights['improvement_suggestions'].append("🔄 Analyze recent performance decline")
            insights['improvement_suggestions'].append("👥 Seek academic counseling")
        
        return insights
    
    except Exception as e:
        logger.error(f"Error getting student insights: {e}")
        return None

@view_bp.route('/view/<usn>')
@login_required
def view_student(usn):
    """Enhanced student view with comprehensive analytics and predictions"""
    try:
        # Load student data
        df = pd.read_csv('data.csv')
        student = df[df['USN'] == int(usn)]
        
        if student.empty:
            flash('Student not found', 'error')
            return redirect(url_for('student.students'))
        
        student_data = student.iloc[0].to_dict()
        
        # Create forms
        viz_form = VisualizationForm()
        
        # Generate insights
        insights = get_student_insights(student_data)
        
        # Create default visualizations
        performance_chart = create_performance_trend_chart(student_data)
        comparison_chart = create_class_comparison_chart(student_data)
        
        # Get prediction if service is available
        prediction_result = None
        if initialize_prediction_service():
            prediction_result = prediction_service.get_prediction_confidence(student_data)
        
        # Calculate additional metrics
        grades = [student_data['sem1'], student_data['sem2'], student_data['sem3'],
                 student_data['sem4'], student_data['sem5'], student_data['sem6']]
        
        analytics = {
            'average_grade': round(np.mean(grades), 2),
            'highest_grade': max(grades),
            'lowest_grade': min(grades),
            'improvement': student_data['sem6'] - student_data['sem1'],
            'consistency': round(np.std(grades), 2),
            'trend': round(np.polyfit(range(len(grades)), grades, 1)[0], 2)
        }
        
        # Get class rank
        df_sorted = df.sort_values('sem6', ascending=False)
        rank = df_sorted.index[df_sorted['USN'] == int(usn)].tolist()[0] + 1
        
        return render_template('view.html',
                             student=student_data,
                             usn=usn,
                             analytics=analytics,
                             insights=insights,
                             prediction_result=prediction_result,
                             performance_chart=performance_chart,
                             comparison_chart=comparison_chart,
                             rank=rank,
                             total_students=len(df),
                             viz_form=viz_form)
    
    except Exception as e:
        logger.error(f"Error in view_student: {e}")
        flash('Error loading student details. Please try again.', 'error')
        return render_template('error.html', error_code=500, error_message="Student view loading failed")

@view_bp.route('/api/visualization/<usn>', methods=['POST'])
@login_required
def generate_visualization(usn):
    """Generate custom visualizations for a student"""
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
        student = df[df['USN'] == int(usn)]
        
        if student.empty:
            return jsonify({'error': 'Student not found'}), 404
        
        student_data = student.iloc[0].to_dict()
        
        # Get form data
        form = VisualizationForm()
        if form.validate_on_submit():
            chart_type = form.chart_type.data
            
            chart_html = None
            
            if chart_type == 'performance_trend':
                chart_html = create_performance_trend_chart(student_data)
            elif chart_type == 'prediction_analysis':
                chart_html = create_prediction_analysis_chart(student_data)
            elif chart_type == 'comparison':
                chart_html = create_class_comparison_chart(student_data)
            elif chart_type == 'distribution':
                chart_html = create_grade_distribution_chart()
            elif chart_type == 'correlation':
                chart_html = create_correlation_matrix()
            
            if chart_html:
                return jsonify({
                    'success': True,
                    'chart_html': chart_html,
                    'chart_type': chart_type
                })
            else:
                return jsonify({'error': 'Failed to generate visualization'}), 500
        
        else:
            return jsonify({'error': 'Form validation failed', 'errors': form.errors}), 400
    
    except Exception as e:
        logger.error(f"Error generating visualization: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@view_bp.route('/api/student/<usn>/comprehensive-report')
@login_required
def get_comprehensive_report(usn):
    """Get comprehensive academic report for a student"""
    try:
        # Load student data
        df = pd.read_csv('data.csv')
        student = df[df['USN'] == int(usn)]
        
        if student.empty:
            return jsonify({'error': 'Student not found'}), 404
        
        student_data = student.iloc[0].to_dict()
        
        # Calculate comprehensive metrics
        grades = [student_data['sem1'], student_data['sem2'], student_data['sem3'],
                 student_data['sem4'], student_data['sem5'], student_data['sem6']]
        
        # Basic statistics
        basic_stats = {
            'average': round(np.mean(grades), 2),
            'median': round(np.median(grades), 2),
            'std_dev': round(np.std(grades), 2),
            'min_grade': min(grades),
            'max_grade': max(grades),
            'range': max(grades) - min(grades)
        }
        
        # Trend analysis
        trend_coeff = np.polyfit(range(len(grades)), grades, 1)[0]
        trend_analysis = {
            'slope': round(trend_coeff, 2),
            'direction': 'improving' if trend_coeff > 0.5 else 'declining' if trend_coeff < -0.5 else 'stable',
            'r_squared': round(np.corrcoef(range(len(grades)), grades)[0, 1]**2, 3)
        }
        
        # Comparative analysis
        class_avg = df[['sem1', 'sem2', 'sem3', 'sem4', 'sem5', 'sem6']].mean().mean()
        percentile = (df['sem6'] < student_data['sem6']).sum() / len(df) * 100
        
        comparative_analysis = {
            'vs_class_avg': round(np.mean(grades) - class_avg, 2),
            'percentile': round(percentile, 1),
            'rank': int((df['sem6'] >= student_data['sem6']).sum())
        }
        
        # Prediction analysis
        prediction_analysis = {}
        if initialize_prediction_service():
            pred_result = prediction_service.get_prediction_confidence(student_data)
            if pred_result:
                prediction_analysis = {
                    'next_semester': round(pred_result['prediction'], 2),
                    'confidence': round(pred_result['confidence'], 3),
                    'expected_change': round(pred_result['prediction'] - grades[-1], 2),
                    'risk_level': 'high' if pred_result['prediction'] < 60 else 'medium' if pred_result['prediction'] < 75 else 'low'
                }
        
        # Insights
        insights = get_student_insights(student_data)
        
        report = {
            'success': True,
            'student': student_data,
            'basic_statistics': basic_stats,
            'trend_analysis': trend_analysis,
            'comparative_analysis': comparative_analysis,
            'prediction_analysis': prediction_analysis,
            'insights': insights,
            'generated_at': datetime.now().isoformat(),
            'generated_by': current_user.email
        }
        
        # Log the action
        logger.info(f"User {current_user.id} generated comprehensive report for USN: {usn}")
        
        return jsonify(report)
    
    except Exception as e:
        logger.error(f"Error generating comprehensive report: {e}")
        return jsonify({'error': 'Internal server error'}), 500