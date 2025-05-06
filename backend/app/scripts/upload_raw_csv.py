import pandas as pd
from sqlalchemy import create_engine

def upload_csv_to_postgres(csv_path: str):
    # Load raw CSV
    df = pd.read_csv(csv_path)
    print(f"📄 Loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")

    # Connect to PostgreSQL
    engine = create_engine("postgresql://postgres:database123@localhost:5432/classforge")
                                
    # Upload to PostgreSQL as 'raw_students' table
    df.to_sql('raw_students', engine, if_exists='replace', index=False)
    print("✅ Raw data inserted into PostgreSQL table: raw_students")

# Example usage
if __name__ == "__main__":
    upload_csv_to_postgres("data/StudentPerformanceFactors.csv")
