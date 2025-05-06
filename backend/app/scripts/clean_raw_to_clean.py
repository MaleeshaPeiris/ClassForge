# this file is to encode student's credentials
import pandas as pd
from sqlalchemy import create_engine

DATABASE_URL = "postgresql://postgres:database123@localhost:5432/classforge"

def clean_and_transfer_raw_to_clean():
    engine = create_engine(DATABASE_URL)

    # Step 1: Load raw data
    df = pd.read_sql("SELECT * FROM raw_students", engine)
    print(f"📄 Loaded {df.shape[0]} raw records from raw_students")

    # Step 2: Drop rows with missing target
    df = df.dropna(subset=["Exam_Score"]).reset_index(drop=True)
    df = df.head(500).copy()

    # Step 3: Map binary features
    binary_cols = ["Extracurricular_Activities", "Internet_Access", "Learning_Disabilities"]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # Step 4: Ordinal encodings
    ordinal = {
        'Parental_Involvement': ['Low', 'Medium', 'High'],
        'Motivation_Level': ['Low', 'Medium', 'High'],
        'Distance_from_Home': ['Near', 'Moderate', 'Far']
    }
    for col, order in ordinal.items():
        df[col] = pd.Categorical(df[col], categories=order, ordered=True).codes

    # Step 5: Peer influence mapping
    df['Peer_Influence_Positive'] = df['Peer_Influence'].map({'Positive': 1, 'Neutral': 0.5, 'Negative': 0})
    df['Peer_Influence_Negative'] = df['Peer_Influence'].map({'Negative': 1, 'Neutral': 0.5, 'Positive': 0})

    # Step 6: School type
    df['school_type_private'] = df['School_Type'].map({'Private': 1, 'Public': 0})

    # Step 7: Select cleaned features
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

    # Step 8: Manually add id column
    cleaned_df.insert(0, 'id', range(1, 1 + len(cleaned_df)))

    # Step 9: Replace students_clean table with proper ID
    cleaned_df.to_sql("students_clean", engine, if_exists="append", index=False) ## use "replace" if want to drop the table

    print("✅ Cleaned data inserted into 'students_clean' with ID column")

if __name__ == "__main__":
    clean_and_transfer_raw_to_clean()






#version 2
# #This will add the backend/app/ folder to the Python path, 
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import pandas as pd
# from sqlalchemy import create_engine
# from models.student_clean import Base  # Ensure schema is respected

# DATABASE_URL = "postgresql://postgres:database123@localhost:5432/classforge"

# def clean_and_transfer_raw_to_clean():
#     # Step 1: Connect to DB
#     engine = create_engine(DATABASE_URL)

#     # Step 2: Load raw data
#     df = pd.read_sql("SELECT * FROM raw_students", engine)
#     print(f"📄 Loaded {df.shape[0]} raw records from raw_students")

#     # Step 3: Drop rows with missing target
#     df = df.dropna(subset=["Exam_Score"]).reset_index(drop=True)
#     df = df.head(500).copy()

#     # Step 4: Map binary features
#     binary_cols = ["Extracurricular_Activities", "Internet_Access", "Learning_Disabilities"]
#     for col in binary_cols:
#         df[col] = df[col].map({"Yes": 1, "No": 0})

#     # Step 5: Ordinal encodings
#     ordinal = {
#         'Parental_Involvement': ['Low', 'Medium', 'High'],
#         'Motivation_Level': ['Low', 'Medium', 'High'],
#         'Distance_from_Home': ['Near', 'Moderate', 'Far']
#     }
#     for col, order in ordinal.items():
#         df[col] = pd.Categorical(df[col], categories=order, ordered=True).codes

#     # Step 6: Peer influence split
#     df['Peer_Influence_Positive'] = df['Peer_Influence'].map({'Positive': 1, 'Neutral': 0.5, 'Negative': 0})
#     df['Peer_Influence_Negative'] = df['Peer_Influence'].map({'Negative': 1, 'Neutral': 0.5, 'Positive': 0})

#     # Step 7: School type mapping
#     df['school_type_private'] = df['School_Type'].map({'Private': 1, 'Public': 0})

#     # Step 8: Select cleaned columns for GAT training
#     selected = df[[
#         'Motivation_Level',
#         'Peer_Influence_Positive',
#         'Peer_Influence_Negative',
#         'Attendance',
#         'Parental_Involvement',
#         'Sleep_Hours',
#         'Extracurricular_Activities',
#         'Tutoring_Sessions',
#         'Internet_Access',
#         'school_type_private',
#         'Exam_Score'
#     ]].copy()

#     # Step 9: Ensure table exists and insert without replacing
#     Base.metadata.create_all(engine)
#     selected.to_sql("students_clean", engine, if_exists="append", index=False)

#     print("✅ Cleaned data successfully inserted into 'students_clean' table")

# if __name__ == "__main__":
#     clean_and_transfer_raw_to_clean()



#version1
# import pandas as pd
# from sqlalchemy import create_engine

# DATABASE_URL = "postgresql://postgres:database123@localhost:5432/classforge"

# def clean_and_transfer_raw_to_clean():
#     # Connect to DB
#     engine = create_engine(DATABASE_URL)

#     # Step 1: Load raw data
#     df = pd.read_sql("SELECT * FROM raw_students", engine)
#     print(f"Loaded {df.shape[0]} raw records")

#     # Step 2: Drop rows with missing target
#     df = df.dropna(subset=["Exam_Score"]).reset_index(drop=True)
#     df = df.head(500).copy()

#     # Step 3: Map binary features
#     binary_cols = ["Extracurricular_Activities", "Internet_Access", "Learning_Disabilities"]
#     for col in binary_cols:
#         df[col] = df[col].map({"Yes": 1, "No": 0})

#     # Step 4: Ordinal encodings
#     ordinal = {
#         'Parental_Involvement': ['Low', 'Medium', 'High'],
#         'Motivation_Level': ['Low', 'Medium', 'High'],
#         'Distance_from_Home': ['Near', 'Moderate', 'Far'],
#     }
#     for col, order in ordinal.items():
#         df[col] = pd.Categorical(df[col], categories=order, ordered=True).codes

#     # Step 5: Peer influence
#     df['Peer_Influence_Positive'] = df['Peer_Influence'].map({'Positive': 1, 'Neutral': 0.5, 'Negative': 0})
#     df['Peer_Influence_Negative'] = df['Peer_Influence'].map({'Negative': 1, 'Neutral': 0.5, 'Positive': 0})

#     # Step 6: School type
#     df['school_type_private'] = df['School_Type'].map({'Private': 1, 'Public': 0})

#     # Step 7: Select features for clean table
#     selected = df[[
#         'Motivation_Level',
#         'Peer_Influence_Positive',
#         'Peer_Influence_Negative',
#         'Attendance',
#         'Parental_Involvement',
#         'Sleep_Hours',
#         'Extracurricular_Activities',
#         'Tutoring_Sessions',
#         'Internet_Access',
#         'school_type_private',
#         'Exam_Score'
#     ]].copy()

#     # Step 8: Insert into students_clean table
#     selected.to_sql("students_clean", engine, if_exists="replace", index=False)
#     print("✅ Cleaned data inserted into 'students_clean' table")

# if __name__ == "__main__":
#     clean_and_transfer_raw_to_clean()
