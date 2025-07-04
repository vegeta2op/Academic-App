from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify
from flask_login import login_required, current_user
from flask_wtf import FlaskForm
from flask_wtf.csrf import validate_csrf
from wtforms import StringField, SubmitField, SelectField, IntegerField
from wtforms.validators import DataRequired, NumberRange, Length
import pandas as pd
import json
import os
import logging
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# Import our advanced prediction service
from services.prediction_service import prediction_service

predictor_bp = Blueprint('predictor', __name__)
logger = logging.getLogger(__name__)

class PredictionForm(FlaskForm):
    model_type = SelectField('Prediction Model', choices=[
        ('ensemble', 'Ensemble Model (Best Accuracy)'),
        ('random_forest', 'Random Forest'),
        ('gradient_boosting', 'Gradient Boosting'),
        ('ridge', 'Ridge Regression'),
        ('lasso', 'Lasso Regression'),
        ('linear', 'Linear Regression')
    ], default='ensemble')
    
    confidence_threshold = SelectField('Confidence Threshold', choices=[
        ('0.8', 'High (80%)'),
        ('0.6', 'Medium (60%)'),
        ('0.4', 'Low (40%)')
    ], default='0.8')
    
    submit = SubmitField('Generate Predictions')

class ManualPredictionForm(FlaskForm):
    name = StringField('Student Name', validators=[DataRequired(), Length(min=2, max=50)])
    sem1 = IntegerField('Semester 1', validators=[DataRequired(), NumberRange(min=0, max=100)])
    sem2 = IntegerField('Semester 2', validators=[DataRequired(), NumberRange(min=0, max=100)])
    sem3 = IntegerField('Semester 3', validators=[DataRequired(), NumberRange(min=0, max=100)])
    sem4 = IntegerField('Semester 4', validators=[DataRequired(), NumberRange(min=0, max=100)])
    sem5 = IntegerField('Semester 5', validators=[DataRequired(), NumberRange(min=0, max=100)])
    model_type = SelectField('Model', choices=[
        ('ensemble', 'Ensemble'),
        ('random_forest', 'Random Forest'),
        ('gradient_boosting', 'Gradient Boosting'),
        ('ridge', 'Ridge Regression')
    ], default='ensemble')
    submit = SubmitField('Predict Performance')

def initialize_prediction_service():
    """Initialize the prediction service with trained models"""
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

def get_prediction_analytics():
    """Get analytics for prediction performance"""
    try:
        if not prediction_service.model_metrics:
            return None
        
        # Get model comparison
        comparison_df = prediction_service.get_model_comparison()
        if comparison_df is None:
            return None
        
        analytics = {
            'best_model': comparison_df.index[0],
            'best_r2': comparison_df['r2'].iloc[0],
            'model_comparison': comparison_df.to_dict('index'),
            'total_models': len(comparison_df),
            'feature_importance': prediction_service.feature_importance.get('ensemble', {})
        }
        
        return analytics
    
    except Exception as e:
        logger.error(f"Error getting prediction analytics: {e}")
        return None

def create_model_comparison_chart():
    """Create a chart comparing different model performances"""
    try:
        if not prediction_service.model_metrics:
            return None
        
        comparison_df = prediction_service.get_model_comparison()
        if comparison_df is None:
            return None
        
        fig = go.Figure()
        
        # Add R² scores
        fig.add_trace(go.Bar(
            x=comparison_df.index,
            y=comparison_df['r2'],
            name='R² Score',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='R² Score',
            template='plotly_white'
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    except Exception as e:
        logger.error(f"Error creating model comparison chart: {e}")
        return None

def create_feature_importance_chart():
    """Create a chart showing feature importance"""
    try:
        if not prediction_service.feature_importance:
            return None
        
        # Get feature importance from the best model
        best_model = 'random_forest'  # Default to random forest
        if best_model in prediction_service.feature_importance:
            importance_dict = prediction_service.feature_importance[best_model]
            
            features = list(importance_dict.keys())
            importance_values = list(importance_dict.values())
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=importance_values,
                y=features,
                orientation='h',
                marker_color='lightgreen'
            ))
            
            fig.update_layout(
                title='Feature Importance',
                xaxis_title='Importance Score',
                yaxis_title='Features',
                template='plotly_white'
            )
            
            return fig.to_html(include_plotlyjs='cdn')
        
        return None
    
    except Exception as e:
        logger.error(f"Error creating feature importance chart: {e}")
        return None

@predictor_bp.route("/predictor")
@login_required
def predictor():
    """Enhanced predictor page with advanced ML models"""
    try:
        # Initialize prediction service
        if not initialize_prediction_service():
            flash('Error initializing prediction models. Please try again.', 'error')
            return render_template('error.html', error_code=500, error_message="Prediction service unavailable")
        
        # Load student data
        df = pd.read_csv('data.csv')
        students = df.to_dict('records')
        
        # Get prediction analytics
        analytics = get_prediction_analytics()
        
        # Create forms
        prediction_form = PredictionForm()
        manual_form = ManualPredictionForm()
        
        # Create visualizations
        model_comparison_chart = create_model_comparison_chart()
        feature_importance_chart = create_feature_importance_chart()
        
        # Get recent predictions (this would be from a database in production)
        recent_predictions = [
            {"student": "John Doe", "predicted_grade": 85.2, "confidence": 0.92, "timestamp": "2 hours ago"},
            {"student": "Jane Smith", "predicted_grade": 78.5, "confidence": 0.88, "timestamp": "1 day ago"}
        ]
        
        return render_template('predictor.html', 
                             students=students,
                             prediction_form=prediction_form,
                             manual_form=manual_form,
                             analytics=analytics,
                             model_comparison_chart=model_comparison_chart,
                             feature_importance_chart=feature_importance_chart,
                             recent_predictions=recent_predictions)
    
    except Exception as e:
        logger.error(f"Predictor error: {e}")
        flash('Error loading predictor page. Please try again.', 'error')
        return render_template('error.html', error_code=500, error_message="Predictor loading failed")

@predictor_bp.route('/predict/batch', methods=['POST'])
@login_required
def batch_predict():
    """Generate predictions for all students"""
    try:
        # Validate CSRF
        csrf_token = request.headers.get('X-CSRFToken')
        if not csrf_token:
            return jsonify({'error': 'CSRF token missing'}), 400
        
        try:
            validate_csrf(csrf_token)
        except:
            return jsonify({'error': 'Invalid CSRF token'}), 400
        
        form = PredictionForm()
        if form.validate_on_submit():
            # Initialize prediction service
            if not initialize_prediction_service():
                return jsonify({'error': 'Prediction service unavailable'}), 500
            
            # Load student data
            df = pd.read_csv('data.csv')
            students = df.to_dict('records')
            
            # Get form data
            model_type = form.model_type.data
            confidence_threshold = float(form.confidence_threshold.data)
            
            # Generate predictions
            predictions = []
            for student in students:
                try:
                    pred_result = prediction_service.get_prediction_confidence(student)
                    if pred_result and pred_result['confidence'] >= confidence_threshold:
                        predictions.append({
                            'name': student['Name'],
                            'usn': student['USN'],
                            'predicted_grade': round(pred_result['prediction'], 2),
                            'confidence': round(pred_result['confidence'], 3),
                            'current_average': round(np.mean([student['sem1'], student['sem2'], student['sem3'], student['sem4'], student['sem5']]), 2),
                            'insights': prediction_service.generate_performance_insights(student)
                        })
                except Exception as e:
                    logger.error(f"Error predicting for student {student.get('Name', 'Unknown')}: {e}")
            
            # Log the action
            logger.info(f"User {current_user.id} generated batch predictions using {model_type}")
            
            return jsonify({
                'success': True,
                'predictions': predictions,
                'total': len(predictions),
                'model_used': model_type,
                'confidence_threshold': confidence_threshold
            })
        
        else:
            return jsonify({'error': 'Form validation failed', 'errors': form.errors}), 400
    
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@predictor_bp.route('/predict/manual', methods=['POST'])
@login_required
def manual_predict():
    """Generate prediction for manually entered student data"""
    try:
        form = ManualPredictionForm()
        if form.validate_on_submit():
            # Initialize prediction service
            if not initialize_prediction_service():
                return jsonify({'error': 'Prediction service unavailable'}), 500
            
            # Prepare student data
            student_data = {
                'Name': form.name.data,
                'sem1': form.sem1.data,
                'sem2': form.sem2.data,
                'sem3': form.sem3.data,
                'sem4': form.sem4.data,
                'sem5': form.sem5.data
            }
            
            # Generate prediction
            pred_result = prediction_service.get_prediction_confidence(student_data)
            
            if pred_result:
                insights = prediction_service.generate_performance_insights(student_data)
                
                # Create performance chart
                performance_chart = prediction_service.create_performance_visualization(
                    student_data, pred_result['prediction']
                )
                
                result = {
                    'success': True,
                    'student': student_data,
                    'prediction': round(pred_result['prediction'], 2),
                    'confidence': round(pred_result['confidence'], 3),
                    'insights': insights,
                    'performance_chart': performance_chart.to_html(include_plotlyjs='cdn') if performance_chart else None,
                    'model_used': form.model_type.data
                }
                
                # Log the action
                logger.info(f"User {current_user.id} generated manual prediction for {form.name.data}")
                
                return jsonify(result)
            
            else:
                return jsonify({'error': 'Failed to generate prediction'}), 500
        
        else:
            return jsonify({'error': 'Form validation failed', 'errors': form.errors}), 400
    
    except Exception as e:
        logger.error(f"Error in manual prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@predictor_bp.route('/api/student/<usn>/detailed-prediction')
@login_required
def get_detailed_prediction(usn):
    """Get detailed prediction analysis for a specific student"""
    try:
        # Load student data
        df = pd.read_csv('data.csv')
        student = df[df['USN'] == int(usn)]
        
        if student.empty:
            return jsonify({'error': 'Student not found'}), 404
        
        student_data = student.iloc[0].to_dict()
        
        # Initialize prediction service
        if not initialize_prediction_service():
            return jsonify({'error': 'Prediction service unavailable'}), 500
        
        # Get detailed prediction
        pred_result = prediction_service.get_prediction_confidence(student_data)
        insights = prediction_service.generate_performance_insights(student_data)
        
        if pred_result:
            # Create performance visualization
            performance_chart = prediction_service.create_performance_visualization(
                student_data, pred_result['prediction']
            )
            
            # Calculate additional metrics
            current_avg = np.mean([student_data['sem1'], student_data['sem2'], 
                                 student_data['sem3'], student_data['sem4'], student_data['sem5']])
            
            predicted_improvement = pred_result['prediction'] - current_avg
            
            result = {
                'success': True,
                'student': student_data,
                'prediction': {
                    'next_semester': round(pred_result['prediction'], 2),
                    'confidence': round(pred_result['confidence'], 3),
                    'confidence_interval': [
                        round(pred_result['prediction'] - pred_result['std_deviation'], 2),
                        round(pred_result['prediction'] + pred_result['std_deviation'], 2)
                    ],
                    'individual_predictions': [round(p, 2) for p in pred_result['individual_predictions']]
                },
                'analytics': {
                    'current_average': round(current_avg, 2),
                    'predicted_improvement': round(predicted_improvement, 2),
                    'performance_trend': 'improving' if predicted_improvement > 0 else 'declining',
                    'risk_level': 'high' if pred_result['prediction'] < 60 else 'medium' if pred_result['prediction'] < 75 else 'low'
                },
                'insights': insights,
                'performance_chart': performance_chart.to_html(include_plotlyjs='cdn') if performance_chart else None
            }
            
            return jsonify(result)
        
        else:
            return jsonify({'error': 'Failed to generate prediction'}), 500
    
    except Exception as e:
        logger.error(f"Error getting detailed prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@predictor_bp.route('/api/retrain-models', methods=['POST'])
@login_required
def retrain_models():
    """Retrain prediction models with latest data"""
    try:
        # Validate CSRF
        csrf_token = request.headers.get('X-CSRFToken')
        if not csrf_token:
            return jsonify({'error': 'CSRF token missing'}), 400
        
        try:
            validate_csrf(csrf_token)
        except:
            return jsonify({'error': 'Invalid CSRF token'}), 400
        
        # Start retraining
        success = prediction_service.train_full_pipeline()
        
        if success:
            logger.info(f"User {current_user.id} initiated model retraining")
            
            # Get updated analytics
            analytics = get_prediction_analytics()
            
            return jsonify({
                'success': True,
                'message': 'Models retrained successfully',
                'analytics': analytics
            })
        else:
            return jsonify({'error': 'Failed to retrain models'}), 500
    
    except Exception as e:
        logger.error(f"Error retraining models: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@predictor_bp.route('/api/model-metrics')
@login_required
def get_model_metrics():
    """Get current model performance metrics"""
    try:
        if not prediction_service.model_metrics:
            return jsonify({'error': 'No model metrics available'}), 404
        
        metrics = prediction_service.model_metrics
        comparison_df = prediction_service.get_model_comparison()
        
        result = {
            'success': True,
            'metrics': metrics,
            'comparison': comparison_df.to_dict('index') if comparison_df is not None else {},
            'best_model': comparison_df.index[0] if comparison_df is not None else 'unknown'
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error getting model metrics: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@predictor_bp.route('/api/export-predictions')
@login_required
def export_predictions():
    """Export predictions for all students"""
    try:
        # Initialize prediction service
        if not initialize_prediction_service():
            return jsonify({'error': 'Prediction service unavailable'}), 500
        
        # Load student data
        df = pd.read_csv('data.csv')
        students = df.to_dict('records')
        
        # Generate predictions for all students
        predictions = []
        for student in students:
            try:
                pred_result = prediction_service.get_prediction_confidence(student)
                if pred_result:
                    predictions.append({
                        'Name': student['Name'],
                        'USN': student['USN'],
                        'Current_Average': round(np.mean([student['sem1'], student['sem2'], student['sem3'], student['sem4'], student['sem5']]), 2),
                        'Predicted_Grade': round(pred_result['prediction'], 2),
                        'Confidence': round(pred_result['confidence'], 3),
                        'Export_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'Exported_By': current_user.email
                    })
            except Exception as e:
                logger.error(f"Error predicting for student {student.get('Name', 'Unknown')}: {e}")
        
        # Convert to DataFrame and then CSV
        predictions_df = pd.DataFrame(predictions)
        csv_data = predictions_df.to_csv(index=False)
        
        logger.info(f"User {current_user.id} exported predictions")
        
        return jsonify({
            'success': True,
            'csv_data': csv_data,
            'filename': f'predictions_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            'total_predictions': len(predictions)
        })
    
    except Exception as e:
        logger.error(f"Error exporting predictions: {e}")
        return jsonify({'error': 'Internal server error'}), 500
