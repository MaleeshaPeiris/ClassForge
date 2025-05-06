from app import create_app, db
from database import init_db

app = create_app()

if __name__ == '__main__':
    init_db(app)
    app.run(debug=True)
