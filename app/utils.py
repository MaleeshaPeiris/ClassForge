import torch
import pandas as pd

def preprocess_data(df):
    df = df.copy()
    df['gender'] = df['gender'].map({'male': 0, 'female': 1})
    df['immigrant_status'] = df['immigrant_status'].astype(int)
    df['SES'] = df['SES'].astype(float)
    df['achievement'] = df['achievement'].astype(float)
    df['psychological_distress'] = df['psychological_distress'].astype(float)
    return df

def allocate_students(df, model, criterion, num_classes):
    if criterion == "academic":
        features = ['achievement']
    elif criterion == "wellbeing":
        features = ['psychological_distress']
    else:
        features = ['achievement', 'psychological_distress']
    
    inputs = torch.tensor(df[features].values, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(inputs)
        predicted_classes = torch.argmax(outputs, dim=1).numpy()

    df['class'] = predicted_classes % num_classes
    return df[['student_id', 'class']]
