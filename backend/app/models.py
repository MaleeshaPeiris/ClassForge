from app import db

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    grade = db.Column(db.String(20))
    wellbeing_score = db.Column(db.Float)

    def serialize(self):
        return {
            "id": self.id,
            "name": self.name,
            "grade": self.grade,
            "wellbeing_score": self.wellbeing_score
        }
