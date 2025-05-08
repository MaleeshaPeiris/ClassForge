from flask import Blueprint, render_template
from . import db  # Import the globally initialized db
from .models import User  # Import the User model

main = Blueprint('main', __name__)

@main.route('/')
def index():
    #  Example of using the database.
    #  It's good practice to do database operations within a request context.
    users = User.query.all()
    return render_template('index.html', users=users)
