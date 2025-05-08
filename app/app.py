# from flask import Flask, request, render_template, jsonify
# import pandas as pd
# import torch
# import boto3
# from model_architecture import StudentClassifier
# from utils import preprocess_data, allocate_students

# app = Flask(__name__)

# # AWS Configuration
# S3_BUCKET = 'class-forge'
# s3 = boto3.client(
#     "s3",
#     aws_access_key_id='YOUR_ACCESS_KEY',
#     aws_secret_access_key='YOUR_SECRET_KEY',
#     region_name='your-region'  # e.g. 'ap-southeast-2'
# )


# # Load model
# #model_path = "model.pth"
# #model = StudentClassifier(input_dim=2, num_classes=3)  # adjust input_dim as per model
# #model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# #model.eval()

# @app.route('/')
# def index():
#     return render_template('index.html')

# """ @app.route('/allocate', methods=['POST'])
# def allocate():
#     file = request.files['file']
#     criterion = request.form['criterion']
#     num_classes = int(request.form['num_classes'])

#     df = pd.read_csv(file)
#     processed_df = preprocess_data(df)
    
#     result = allocate_students(processed_df, model, criterion, num_classes)
#     return jsonify(result.to_dict(orient='records')) """

# @app.route('/allocate', methods=['POST'])
# def allocate_students():
#     if 'file' not in request.files:
#         return "No file uploaded", 400
    
#     print(boto3.__version__)

#     file = request.files['file']
#     df = pd.read_csv(file)

#     # 🔁 MOCK: pretend model gives random class assignments
#     df['allocated_class'] = (df.index % 3) + 1  # Dummy 3-class rotation

#     # Return the result to frontend (e.g., display as table or chart)
#     return jsonify(df.to_dict(orient='records'))


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
