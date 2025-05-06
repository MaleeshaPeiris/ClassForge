import pandas as pd
from sqlalchemy import create_engine

DATABASE_URL = "postgresql://postgres:database123@localhost:5432/classforge"

def load_clean_features():
    engine = create_engine(DATABASE_URL)

    # Read from students_clean table
    df = pd.read_sql("SELECT * FROM students_clean ORDER BY id", engine)

    # Drop ID column for features
    X = df.drop(columns=["id", "Exam_Score"]).values
    y = df["Exam_Score"].values

    return X, y




## version 1
# import pandas as pd
# from sqlalchemy import create_engine

# def load_student_features():
#     engine = create_engine("postgresql://user:password@localhost:5432/classforge")
#     query = "SELECT * FROM students"
#     df = pd.read_sql(query, engine)
#     return df
