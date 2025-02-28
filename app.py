# app.py
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash
from models import db, Issue
import os
"""
app.py
This module contains the main application code for the Tenant Issue Tracker web application.
It uses the Flask framework to handle web requests and SQLAlchemy for database interactions.
Routes:
    /: Renders the index page where users can submit new issues.
    /submit: Handles the form submission for new issues. Validates input, handles image uploads, and saves the issue to the database.
    /dashboard: Renders the dashboard page displaying all submitted issues.
Configuration:
    SECRET_KEY: Secret key for session management and CSRF protection.
    SQLALCHEMY_DATABASE_URI: URI for the SQLite database.
    SQLALCHEMY_TRACK_MODIFICATIONS: Flag to disable SQLAlchemy modification tracking.
    UPLOAD_FOLDER: Directory for storing uploaded images.
Functions:
    index(): Renders the index.html template.
    submit_issue(): Handles the submission of new issues, including validation, image upload, and database insertion.
    dashboard(): Renders the dashboard.html template with a list of all issues.
Usage:
    Run this module directly to start the Flask development server.
"""


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///issues.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'

db.init_app(app)

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_issue():
    text = request.form.get('description')
    image = request.files.get('image')
    
    # Basic validation
    if not text:
        flash('Please provide a description of the issue')
        return redirect(url_for('index'))
    
    # Handle image upload
    image_path = None
    if image and image.filename:
        filename = f"{str(uuid.uuid4())}{os.path.splitext(image.filename)[1]}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)
    
    # Create new issue
    new_issue = Issue(text=text, image_path=image_path)
    
    # Here we'll later add the categorization logic
    # new_issue.category = classify_issue(text)
    # new_issue.urgency = determine_urgency(text, image_path)
    
    db.session.add(new_issue)
    db.session.commit()
    
    flash('Issue submitted successfully!')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    issues = Issue.query.order_by(Issue.timestamp.desc()).all()
    return render_template('dashboard.html', issues=issues)

if __name__ == '__main__':
    app.run(debug=True)