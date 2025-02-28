from flask_sqlalchemy import SQLAlchemy
"""
This module defines the database models for the tenant issue tracker application.
Classes:
    Issue: A class representing an issue reported by a tenant.
Attributes:
    id (str): The unique identifier for the issue.
    text (str): The description of the issue.
    image_path (str, optional): The file path to an image related to the issue.
    timestamp (datetime): The date and time when the issue was reported.
    category (str, optional): The category of the issue.
    urgency (str, optional): The urgency level of the issue.
    status (str): The current status of the issue.
Methods:
    __init__(self, text, image_path=None): Initializes a new instance of the Issue class.
"""
from datetime import datetime
import uuid

db = SQLAlchemy()

class Issue(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    text = db.Column(db.Text, nullable=False)
    image_path = db.Column(db.String(255), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    category = db.Column(db.String(50), nullable=True)
    urgency = db.Column(db.String(20), nullable=True)
    status = db.Column(db.String(20), default="new")
    
    def __init__(self, text, image_path=None):
        self.id = str(uuid.uuid4())  # Anonymous identifier
        self.text = text
        self.image_path = image_path