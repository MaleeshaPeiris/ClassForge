# app/utils/csv_handler.py

import pandas as pd
from app.models.model import RawStudentData, TransformedStudentData
from app import db
from app.models.model import TransformedStudentData
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app import db
from app.models.model import RawStudentData
import pandas as pd

def upload_raw_csv(filepath: str):
    df = pd.read_csv(filepath)

    inserted = 0
    skipped = 0

    for _, row in df.iterrows():
        existing = RawStudentData.query.filter_by(student_id=row['student_id']).first()
        if existing:
            skipped += 1
            continue

        student = RawStudentData(
            student_id=row['student_id'],
            gender=row['gender'],
            immigrant_status=row['immigrant_status'],
            SES=row['SES'],
            achievement=row['achievement'],
            psychological_distress=row['psychological_distress']
        )
        db.session.add(student)
        inserted += 1

    db.session.commit()
    print(f"✅ Upload complete: {inserted} new rows inserted, {skipped} skipped (duplicates).")




# Preprocess raw data to transformed data
# This function will be called automatically after uploading the CSV
def preprocess_raw_to_transformed():
    raw_students = RawStudentData.query.all()
    if not raw_students:
        print("❌ No raw student data found.")
        return

    # Build DataFrame
    df = pd.DataFrame([{
        "student_id": str(s.student_id),  # 👈 Ensure it's a string
        "gender": s.gender,
        "immigrant_status": s.immigrant_status,
        "SES": s.SES,
        "achievement": s.achievement,
        "psychological_distress": s.psychological_distress
    } for s in raw_students])

    # Encode categorical
    df['encoded_gender'] = LabelEncoder().fit_transform(df['gender'])
    df['encoded_immigrant_status'] = LabelEncoder().fit_transform(df['immigrant_status'])

    # Normalize continuous
    scaler = StandardScaler()
    df[['SES', 'achievement', 'psychological_distress']] = scaler.fit_transform(
        df[['SES', 'achievement', 'psychological_distress']]
    )

    # Insert into TransformedStudentData
    for _, row in df.iterrows():
        transformed = TransformedStudentData(
            student_id=row['student_id'],  # 👈 Still string
            encoded_gender=row['encoded_gender'],
            encoded_immigrant_status=row['encoded_immigrant_status'],
            ses=row['SES'],
            achievement=row['achievement'],
            psychological_distress=row['psychological_distress']
        )
        db.session.add(transformed)

    db.session.commit()
    print(f"✅ Preprocessing complete: {len(df)} records transformed.")








#version 1.0.0
# def preprocess_raw_to_transformed():
#     raw_students = RawStudentData.query.all()
#     if not raw_students:
#         print("❌ No raw student data found.")
#         return

#     df = pd.DataFrame([{
#         "student_id": s.student_id,
#         "gender": s.gender,
#         "immigrant_status": s.school_type,
#         "SES": float(s.parental_education),
#         "achievement": float(s.exam_score),
#         "psychological_distress": 0.0  # Add this field if your dataset supports it
#     } for s in raw_students])

#     # Encode categorical features
#     gender_enc = LabelEncoder()
#     immigrant_enc = LabelEncoder()
#     df['encoded_gender'] = gender_enc.fit_transform(df['gender'])
#     df['encoded_immigrant_status'] = immigrant_enc.fit_transform(df['immigrant_status'])

#     # Normalize continuous features
#     scaler = StandardScaler()
#     df[['SES', 'achievement', 'psychological_distress']] = scaler.fit_transform(
#         df[['SES', 'achievement', 'psychological_distress']]
#     )

#     for _, row in df.iterrows():
#         transformed = TransformedStudentData(
#             student_id=row['student_id'],
#             encoded_gender=row['encoded_gender'],
#             encoded_immigrant_status=row['encoded_immigrant_status'],
#             ses=row['SES'],
#             achievement=row['achievement'],
#             psychological_distress=row['psychological_distress']
#         )
#         db.session.add(transformed)

#     db.session.commit()
#     print("✅ Transformed data inserted with scaled features.")