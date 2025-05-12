from flask import Flask, request, render_template, jsonify
import pandas as pd
import torch
from utils import process_csvfile, train_model_from_csv, GAT
from sklearn.cluster import KMeans


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/allocate', methods=['POST'])
def allocate_students():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    train_dataset_df = pd.read_csv("normalized_data_with_labels.csv")
    file = request.files['file']
    df = pd.read_csv(file)
    criterion = request.form['criterion']
    num_classes = int(request.form['num_classes'])
    global current_model
    current_model = train_model_from_csv(train_dataset_df,num_classes,criterion) 
    data = process_csvfile(df, criterion)
    with torch.no_grad():
        out = current_model(data.x, data.edge_index)
        df['allocated_class'] = torch.argmax(out, dim=1).tolist()
    return  jsonify(df.to_dict(orient='records'))


current_model = None


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)