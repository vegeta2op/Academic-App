from flask import Blueprint, render_template, request, jsonify, send_file, session
from flask_login import login_required, current_user
from flask_wtf import FlaskForm
from flask_wtf.csrf import validate_csrf
from wtforms import SelectField, StringField, TextAreaField, SubmitField, DateField
from wtforms.validators import DataRequired, Length
import pandas as pd
import json
import os
import logging
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import base64

reports_bp = Blueprint('reports', __name__)
logger = logging.getLogger(__name__)

class ReportForm(FlaskForm):
    report_type = SelectField('Report Type', choices=[
        ('performance_summary', 'Performance Summary'),
        ('student_progress', 'Student Progress Report'),
        ('class_analytics', 'Class Analytics Report'),
        ('prediction_analysis', 'Prediction Analysis Report'),
        ('semester_comparison', 'Semester Comparison Report'),
        ('grade_distribution', 'Grade Distribution Report'),
        ('top_performers', 'Top Performers Report'),
        ('at_risk_students', 'At-Risk Students Report'),
        ('improvement_trends', 'Improvement Trends Report'),
        ('comprehensive', 'Comprehensive Report')
    ], validators=[DataRequired()])
    
    format_type = SelectField('Export Format', choices=[
        ('html', 'HTML Report'),
        ('pdf', 'PDF Report'),
        ('csv', 'CSV Data'),
        ('json', 'JSON Data'),
        ('excel', 'Excel Spreadsheet')
    ], default='html', validators=[DataRequired()])
    
    date_range = SelectField('Date Range', choices=[
        ('current_semester', 'Current Semester'),
        ('last_semester', 'Last Semester'),
        ('academic_year', 'Academic Year'),
        ('all_time', 'All Time'),
        ('custom', 'Custom Range')
    ], default='current_semester')
    
    start_date = DateField('Start Date', format='%Y-%m-%d')
    end_date = DateField('End Date', format='%Y-%m-%d')
    
    include_charts = SelectField('Include Charts', choices=[
        ('yes', 'Yes'),
        ('no', 'No')
    ], default='yes')
    
    include_recommendations = SelectField('Include AI Recommendations', choices=[
        ('yes', 'Yes'),
        ('no', 'No')
    ], default='yes')
    
    custom_filters = TextAreaField('Custom Filters (JSON)', 
                                  description='Advanced filters in JSON format',
                                  render_kw={'rows': 3})
    
    report_title = StringField('Report Title', 
                              default='Academic Performance Report',
                              validators=[Length(max=100)])
    
    report_description = TextAreaField('Report Description', 
                                      render_kw={'rows': 2})
    
    submit = SubmitField('Generate Report')

def load_student_data():
    """Load and prepare student data"""
    try:
        df = pd.read_csv('data.csv')
        return df
    except Exception as e:
        logger.error(f"Error loading student data: {e}")
        return None

def calculate_comprehensive_stats(df):
    """Calculate comprehensive statistics for reports"""
    try:
        semesters = ['sem1', 'sem2', 'sem3', 'sem4', 'sem5', 'sem6']
        available_sems = [sem for sem in semesters if sem in df.columns]
        
        stats = {
            'total_students': len(df),
            'average_performance': df[available_sems].mean().mean(),
            'median_performance': df[available_sems].median().median(),
            'std_deviation': df[available_sems].std().mean(),
            'min_grade': df[available_sems].min().min(),
            'max_grade': df[available_sems].max().max(),
            'pass_rate': (df[available_sems] >= 60).mean().mean() * 100,
            'distinction_rate': (df[available_sems] >= 85).mean().mean() * 100,
            'fail_rate': (df[available_sems] < 60).mean().mean() * 100
        }
        
        # Calculate trends
        if len(available_sems) > 1:
            first_sem = df[available_sems[0]].mean()
            last_sem = df[available_sems[-1]].mean()
            stats['trend'] = 'improving' if last_sem > first_sem else 'declining'
            stats['trend_percentage'] = ((last_sem - first_sem) / first_sem) * 100
        else:
            stats['trend'] = 'stable'
            stats['trend_percentage'] = 0
        
        return stats
    except Exception as e:
        logger.error(f"Error calculating stats: {e}")
        return {}

def generate_performance_charts(df):
    """Generate performance charts for reports"""
    try:
        charts = {}
        semesters = ['sem1', 'sem2', 'sem3', 'sem4', 'sem5', 'sem6']
        available_sems = [sem for sem in semesters if sem in df.columns]
        
        # Performance trends chart
        if len(available_sems) > 1:
            averages = [df[sem].mean() for sem in available_sems]
            sem_labels = [f'Sem {i+1}' for i in range(len(available_sems))]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sem_labels,
                y=averages,
                mode='lines+markers',
                name='Average Performance',
                line=dict(color='#3498db', width=4),
                marker=dict(size=10)
            ))
            
            fig.update_layout(
                title='Performance Trends Over Semesters',
                xaxis_title='Semester',
                yaxis_title='Average Grade',
                template='plotly_white'
            )
            
            charts['performance_trends'] = fig
        
        # Grade distribution chart
        overall_grades = df[available_sems].mean(axis=1)
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=overall_grades,
            nbinsx=15,
            name='Grade Distribution',
            marker_color='#2ecc71'
        ))
        
        fig.update_layout(
            title='Grade Distribution',
            xaxis_title='Average Grade',
            yaxis_title='Number of Students',
            template='plotly_white'
        )
        
        charts['grade_distribution'] = fig
        
        # Top performers chart
        df_copy = df.copy()
        df_copy['average'] = df_copy[available_sems].mean(axis=1)
        top_10 = df_copy.nlargest(10, 'average')
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_10['average'],
            y=top_10['Name'],
            orientation='h',
            name='Top Performers',
            marker_color='#f39c12'
        ))
        
        fig.update_layout(
            title='Top 10 Performers',
            xaxis_title='Average Grade',
            yaxis_title='Student Name',
            template='plotly_white'
        )
        
        charts['top_performers'] = fig
        
        return charts
    except Exception as e:
        logger.error(f"Error generating charts: {e}")
        return {}

def generate_ai_recommendations(df, stats):
    """Generate AI-powered recommendations"""
    try:
        recommendations = []
        
        # Performance-based recommendations
        if stats.get('average_performance', 0) < 70:
            recommendations.append({
                'type': 'warning',
                'title': 'Low Overall Performance',
                'description': f'Class average is {stats["average_performance"]:.1f}%. Consider implementing additional support measures.',
                'action': 'Implement tutoring programs and additional practice sessions'
            })
        
        if stats.get('pass_rate', 0) < 80:
            recommendations.append({
                'type': 'critical',
                'title': 'Low Pass Rate',
                'description': f'Only {stats["pass_rate"]:.1f}% of students are passing. Immediate intervention required.',
                'action': 'Identify struggling students and provide intensive support'
            })
        
        if stats.get('std_deviation', 0) > 15:
            recommendations.append({
                'type': 'info',
                'title': 'High Performance Variance',
                'description': 'Large performance gaps between students detected.',
                'action': 'Consider differentiated instruction approaches'
            })
        
        # Trend-based recommendations
        if stats.get('trend') == 'declining':
            recommendations.append({
                'type': 'warning',
                'title': 'Declining Performance Trend',
                'description': f'Performance has declined by {abs(stats["trend_percentage"]):.1f}% over semesters.',
                'action': 'Review curriculum and teaching methods'
            })
        
        # Default positive recommendation
        if not recommendations:
            recommendations.append({
                'type': 'success',
                'title': 'Strong Performance',
                'description': 'Students are performing well overall. Continue current strategies.',
                'action': 'Maintain current teaching methods and consider advanced challenges'
            })
        
        return recommendations
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return []

def create_pdf_report(data, charts, recommendations):
    """Create PDF report"""
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,
            textColor=colors.HexColor('#2c3e50')
        )
        story.append(Paragraph(data.get('report_title', 'Academic Performance Report'), title_style))
        story.append(Spacer(1, 20))
        
        # Summary statistics
        summary_data = [
            ['Metric', 'Value'],
            ['Total Students', str(data['stats']['total_students'])],
            ['Average Performance', f"{data['stats']['average_performance']:.1f}%"],
            ['Pass Rate', f"{data['stats']['pass_rate']:.1f}%"],
            ['Distinction Rate', f"{data['stats']['distinction_rate']:.1f}%"]
        ]
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 30))
        
        # Recommendations
        if recommendations:
            story.append(Paragraph("AI Recommendations", styles['Heading1']))
            story.append(Spacer(1, 10))
            
            for rec in recommendations:
                story.append(Paragraph(f"• {rec['title']}: {rec['description']}", styles['Normal']))
                story.append(Spacer(1, 10))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        logger.error(f"Error creating PDF report: {e}")
        return None

def export_to_csv(df, stats):
    """Export data to CSV"""
    try:
        # Create comprehensive export
        export_data = df.copy()
        
        # Add calculated fields
        semesters = ['sem1', 'sem2', 'sem3', 'sem4', 'sem5', 'sem6']
        available_sems = [sem for sem in semesters if sem in df.columns]
        
        export_data['Average_Grade'] = export_data[available_sems].mean(axis=1)
        export_data['Best_Performance'] = export_data[available_sems].max(axis=1)
        export_data['Worst_Performance'] = export_data[available_sems].min(axis=1)
        export_data['Performance_Std'] = export_data[available_sems].std(axis=1)
        export_data['Pass_Status'] = export_data['Average_Grade'] >= 60
        export_data['Distinction_Status'] = export_data['Average_Grade'] >= 85
        export_data['At_Risk'] = export_data['Average_Grade'] < 60
        
        # Calculate trends
        if len(available_sems) > 1:
            export_data['Trend'] = export_data[available_sems[-1]] - export_data[available_sems[0]]
            export_data['Trend_Status'] = export_data['Trend'].apply(
                lambda x: 'Improving' if x > 0 else 'Declining' if x < 0 else 'Stable'
            )
        
        return export_data.to_csv(index=False)
        
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")
        return None

def export_to_json(df, stats, recommendations):
    """Export data to JSON"""
    try:
        export_data = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'generated_by': current_user.email if current_user.is_authenticated else 'System',
                'total_students': len(df),
                'report_type': 'comprehensive'
            },
            'statistics': stats,
            'recommendations': recommendations,
            'student_data': df.to_dict('records')
        }
        
        return json.dumps(export_data, indent=2)
        
    except Exception as e:
        logger.error(f"Error exporting to JSON: {e}")
        return None

@reports_bp.route('/reports')
@login_required
def reports():
    """Main reports page"""
    try:
        # Load student data
        df = load_student_data()
        if df is None:
            return render_template('error.html', error_code=500, error_message="Unable to load student data")
        
        # Calculate basic stats
        stats = calculate_comprehensive_stats(df)
        
        # Get recent reports (in production, this would come from database)
        recent_reports = [
            {
                'id': 1,
                'title': 'Monthly Performance Report',
                'type': 'performance_summary',
                'format': 'HTML',
                'generated_at': '2024-01-15 10:30:00',
                'generated_by': 'admin@academic.com',
                'status': 'completed'
            },
            {
                'id': 2,
                'title': 'Top Performers Analysis',
                'type': 'top_performers',
                'format': 'PDF',
                'generated_at': '2024-01-14 15:45:00',
                'generated_by': 'admin@academic.com',
                'status': 'completed'
            }
        ]
        
        # Create form
        form = ReportForm()
        
        return render_template('reports.html', 
                             form=form, 
                             stats=stats, 
                             recent_reports=recent_reports)
    
    except Exception as e:
        logger.error(f"Reports error: {e}")
        return render_template('error.html', error_code=500, error_message="Reports loading failed")

@reports_bp.route('/generate-report', methods=['POST'])
@login_required
def generate_report():
    """Generate a report based on form data"""
    try:
        form = ReportForm()
        if form.validate_on_submit():
            # Load data
            df = load_student_data()
            if df is None:
                return jsonify({'error': 'Unable to load student data'}), 500
            
            # Calculate stats
            stats = calculate_comprehensive_stats(df)
            
            # Generate charts if requested
            charts = {}
            if form.include_charts.data == 'yes':
                charts = generate_performance_charts(df)
            
            # Generate recommendations if requested
            recommendations = []
            if form.include_recommendations.data == 'yes':
                recommendations = generate_ai_recommendations(df, stats)
            
            # Prepare report data
            report_data = {
                'report_title': form.report_title.data,
                'report_description': form.report_description.data,
                'report_type': form.report_type.data,
                'format_type': form.format_type.data,
                'generated_at': datetime.now().isoformat(),
                'generated_by': current_user.email if current_user.is_authenticated else 'System',
                'stats': stats,
                'charts': charts,
                'recommendations': recommendations,
                'student_data': df.to_dict('records')
            }
            
            # Export based on format
            if form.format_type.data == 'html':
                return render_template('report_output.html', data=report_data)
            
            elif form.format_type.data == 'pdf':
                pdf_buffer = create_pdf_report(report_data, charts, recommendations)
                if pdf_buffer:
                    return send_file(
                        pdf_buffer,
                        mimetype='application/pdf',
                        as_attachment=True,
                        download_name=f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
                    )
                else:
                    return jsonify({'error': 'PDF generation failed'}), 500
            
            elif form.format_type.data == 'csv':
                csv_data = export_to_csv(df, stats)
                if csv_data:
                    return jsonify({
                        'success': True,
                        'format': 'csv',
                        'data': csv_data,
                        'filename': f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                    })
                else:
                    return jsonify({'error': 'CSV export failed'}), 500
            
            elif form.format_type.data == 'json':
                json_data = export_to_json(df, stats, recommendations)
                if json_data:
                    return jsonify({
                        'success': True,
                        'format': 'json',
                        'data': json_data,
                        'filename': f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                    })
                else:
                    return jsonify({'error': 'JSON export failed'}), 500
            
            else:
                return jsonify({'error': 'Unsupported format'}), 400
        
        else:
            return jsonify({'error': 'Form validation failed', 'errors': form.errors}), 400
    
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@reports_bp.route('/api/report-templates')
@login_required
def get_report_templates():
    """Get available report templates"""
    try:
        templates = {
            'performance_summary': {
                'name': 'Performance Summary',
                'description': 'Overall academic performance statistics and trends',
                'fields': ['total_students', 'average_performance', 'pass_rate', 'trends'],
                'charts': ['performance_trends', 'grade_distribution'],
                'estimated_time': '2-3 minutes'
            },
            'student_progress': {
                'name': 'Student Progress Report',
                'description': 'Individual student progress tracking and analysis',
                'fields': ['individual_grades', 'progress_trends', 'recommendations'],
                'charts': ['individual_performance', 'comparison_charts'],
                'estimated_time': '3-5 minutes'
            },
            'class_analytics': {
                'name': 'Class Analytics',
                'description': 'Comprehensive class performance analytics',
                'fields': ['class_statistics', 'performance_distribution', 'insights'],
                'charts': ['multiple_visualizations', 'correlation_analysis'],
                'estimated_time': '5-7 minutes'
            },
            'comprehensive': {
                'name': 'Comprehensive Report',
                'description': 'All-inclusive report with detailed analysis',
                'fields': ['everything'],
                'charts': ['all_charts'],
                'estimated_time': '10-15 minutes'
            }
        }
        
        return jsonify({'success': True, 'templates': templates})
    
    except Exception as e:
        logger.error(f"Error getting report templates: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@reports_bp.route('/api/report-preview')
@login_required
def preview_report():
    """Preview report before generation"""
    try:
        report_type = request.args.get('type', 'performance_summary')
        
        # Load sample data
        df = load_student_data()
        if df is None:
            return jsonify({'error': 'Unable to load data'}), 500
        
        # Calculate preview stats
        stats = calculate_comprehensive_stats(df)
        
        # Create preview
        preview = {
            'report_type': report_type,
            'estimated_pages': 5,
            'estimated_time': '3-5 minutes',
            'included_sections': [
                'Executive Summary',
                'Performance Statistics',
                'Visual Analytics',
                'AI Recommendations',
                'Detailed Data'
            ],
            'sample_stats': {
                'total_students': stats.get('total_students', 0),
                'average_performance': round(stats.get('average_performance', 0), 1),
                'pass_rate': round(stats.get('pass_rate', 0), 1)
            }
        }
        
        return jsonify({'success': True, 'preview': preview})
    
    except Exception as e:
        logger.error(f"Error creating report preview: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@reports_bp.route('/api/reports/history')
@login_required
def get_report_history():
    """Get report generation history"""
    try:
        # In production, this would come from database
        history = [
            {
                'id': 1,
                'title': 'Monthly Performance Report',
                'type': 'performance_summary',
                'format': 'HTML',
                'generated_at': '2024-01-15 10:30:00',
                'generated_by': 'admin@academic.com',
                'status': 'completed',
                'file_size': '2.5 MB',
                'download_count': 12
            },
            {
                'id': 2,
                'title': 'Top Performers Analysis',
                'type': 'top_performers',
                'format': 'PDF',
                'generated_at': '2024-01-14 15:45:00',
                'generated_by': 'admin@academic.com',
                'status': 'completed',
                'file_size': '1.8 MB',
                'download_count': 8
            }
        ]
        
        return jsonify({'success': True, 'history': history})
    
    except Exception as e:
        logger.error(f"Error getting report history: {e}")
        return jsonify({'error': 'Internal server error'}), 500 