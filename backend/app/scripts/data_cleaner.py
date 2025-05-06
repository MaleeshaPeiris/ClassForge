# scripts/data_cleaner.py

import pandas as pd
from sqlalchemy import create_engine

DATABASE_URL = "postgresql://postgres:database123@localhost:5432/classforge"

def clean_and_store_raw_students(limit=500, replace=True):
    engine = create_engine(DATABASE_URL)

    # Load from raw_students table
    df = pd.read_sql("SELECT * FROM raw_students", engine)
    print(f"📄 Loaded {df.shape[0]} raw records")

    # Drop rows missing the target value
    df = df.dropna(subset=["Exam_Score"]).reset_index(drop=True)
    df = df.head(limit).copy()

    # Map binary yes/no columns to 1/0
    binary_cols = ["Extracurricular_Activities", "Internet_Access", "Learning_Disabilities"]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # Ordinal categorical encoding
    ordinal = {
        'Parental_Involvement': ['Low', 'Medium', 'High'],
        'Motivation_Level': ['Low', 'Medium', 'High'],
        'Distance_from_Home': ['Near', 'Moderate', 'Far']
    }
    for col, order in ordinal.items():
        df[col] = pd.Categorical(df[col], categories=order, ordered=True).codes

    # Peer influence numerical features
    df['Peer_Influence_Positive'] = df['Peer_Influence'].map({'Positive': 1, 'Neutral': 0.5, 'Negative': 0})
    df['Peer_Influence_Negative'] = df['Peer_Influence'].map({'Negative': 1, 'Neutral': 0.5, 'Positive': 0})

    # School type binary
    df['school_type_private'] = df['School_Type'].map({'Private': 1, 'Public': 0})

    # Select 10 cleaned features + target
    cleaned_df = df[[
        'Motivation_Level',
        'Peer_Influence_Positive',
        'Peer_Influence_Negative',
        'Attendance',
        'Parental_Involvement',
        'Sleep_Hours',
        'Extracurricular_Activities',
        'Tutoring_Sessions',
        'Internet_Access',
        'school_type_private',
        'Exam_Score'
    ]].copy()

    # Add manual ID column
    cleaned_df.insert(0, 'id', range(1, 1 + len(cleaned_df)))

    # Store into students_clean
    mode = "replace" if replace else "append"
    cleaned_df.to_sql("students_clean", engine, if_exists=mode, index=False)

    print(f"✅ Cleaned data {'replaced' if replace else 'appended to'} 'students_clean' with {len(cleaned_df)} records")

if __name__ == "__main__":
    clean_and_store_raw_students()
