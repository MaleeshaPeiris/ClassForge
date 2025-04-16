from flask import Blueprint, jsonify
from app.models import Student

main = Blueprint('main', __name__)

@main.route('/api/students')
def get_students():
    students = Student.query.all()
    return jsonify([s.serialize() for s in students])
