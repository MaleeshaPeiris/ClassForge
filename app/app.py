from flask import Flask, request, render_template, jsonify, redirect
import pandas as pd
import torch
from utils import process_csvfile, train_model_from_csv, GAT
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import io
from flask import send_file, url_for
import networkx as nx
import matplotlib.cm as cm
import os
from pyvis.network import Network


app = Flask(__name__)
G = None  # Global variable to store the graph

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/allocate', methods=['POST'])
def allocate_students():
    global G
    if 'file' not in request.files:
        return "No file uploaded", 400

    # Read form inputs
    file = request.files['file']

    num_classes = int(request.form.get('num_classes', 3))

    # Read feature weights (as percentages)
    academic_weight = int(request.form.get('academic_weight', 50))
    wellbeing_weight = int(request.form.get('wellbeing_weight', 50))

    # Load training dataset and uploaded file
    train_dataset_df = pd.read_csv("student_dataset.csv")
    df = pd.read_csv(file)

    # Train the model using the given weights
    global current_model
    current_model = train_model_from_csv(
        train_dataset_df, num_classes, 
        academic_weight,wellbeing_weight
    )

    # Preprocess uploaded data using the same weights
    data, G = process_csvfile(
        df, academic_weight,wellbeing_weight
    )

    # Predict and allocate classes
    with torch.no_grad():
        out = current_model(data.x, data.edge_index)
        df['allocated_class'] = torch.argmax(out, dim=1).tolist() 
        print("Allocated classes:", df['allocated_class'].unique())
        print(df)
        df['random_label'] = np.random.randint(0, num_classes, size=len(df))

    for i in range(len(df)):
        G.nodes[i]['random_label'] = df.loc[i, 'random_label']
        G.nodes[i]['allocated_class'] = df.loc[i, 'allocated_class']
        G.nodes[i]['gender_code'] = df.loc[i, 'gender_code']

    url1 = graph_image()
    url2 = graph_image2()

    return jsonify({
        "students": df.to_dict(orient='records'),
        "graph_image_url": url1,
        "graph_image2_url": url2
    })




@app.route('/interactive_graph')
def interactive_graph():
    global G
    if G is None:
        return "No graph available. Please allocate students first.", 400
    
    for n, data in G.nodes(data=True):
        for key in data:
            val = data[key]
            if isinstance(val, torch.Tensor):
                if val.numel() == 1:
                    data[key] = val.item()  # Convert 1-element tensor to scalar
                else:
                    data[key] = val.tolist()  
            elif isinstance(val, (np.integer, np.int64, np.int32)):
                data[key] = int(val)
            elif isinstance(val, (np.floating, np.float32, np.float64)):
                data[key] = float(val)
            elif isinstance(val, np.bool_):
                data[key] = bool(val)

    # Create a Pyvis network
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")

    # Convert NetworkX graph to Pyvis network
    net.from_nx(G)

    # Optional: color nodes by class
    for node in net.nodes:
        cls = G.nodes[node['id']].get('allocated_class', None)
        if cls is not None:
            node['color'] = f"hsl({(cls * 40) % 360}, 80%, 60%)"

    # Save to file
    graph_path = os.path.join("static", "interactive_graph.html")
    net.save_graph(graph_path)

    # Return the path (you can redirect or use this in frontend)
    return redirect(url_for('static', filename='interactive_graph.html'))



def graph_image():
    global G
    if G is None:
        return jsonify({"error": "No graph available. Please upload and allocate first."}), 400

    # Get number of unique classes
    allocated_classes = [data['allocated_class'] for _, data in G.nodes(data=True)]
    unique_classes = sorted(set(allocated_classes))
    class_to_color = {cls: i for i, cls in enumerate(unique_classes)}

    # Use a colormap
    cmap = cm.get_cmap('tab10', len(unique_classes))
    color_map = [cmap(class_to_color[G.nodes[n]['allocated_class']]) for n in G.nodes]

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=False, node_size=45, node_color=color_map, edge_color='gray', ax=ax)

    # Save image to static/images directory
    image_filename = "graph_image.png"
    image_path = os.path.join("static", "images", image_filename)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    plt.savefig(image_path)
    plt.close(fig)

    # Return the URL for the saved image
    image_url = url_for('static', filename=f'images/{image_filename}', _external=False)
    return image_url


def graph_image2():
    global G
    if G is None:
        return jsonify({"error": "No graph available. Please upload and allocate first."}), 400

    # Get number of unique classes
    allocated_classes = [data['random_label'] for _, data in G.nodes(data=True)]
    unique_classes = sorted(set(allocated_classes))
    class_to_color = {cls: i for i, cls in enumerate(unique_classes)}

    # Use a colormap
    cmap = cm.get_cmap('tab10', len(unique_classes))
    color_map = [cmap(class_to_color[G.nodes[n]['random_label']]) for n in G.nodes]

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=False, node_size=45, node_color=color_map, edge_color='gray', ax=ax)

    # Save image to static/images directory
    image_filename = "graph2_image.png"
    image_path = os.path.join("static", "images", image_filename)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    plt.savefig(image_path)
    plt.close(fig)

    # Return the URL for the saved image
    image2_url = url_for('static', filename=f'images/{image_filename}', _external=False)
    return image2_url


current_model = None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
