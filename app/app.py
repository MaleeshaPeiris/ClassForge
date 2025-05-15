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
    unique_allocated_classes = sorted(int(c) for c in df['allocated_class'].unique())
    unique_random_allocated_classes = sorted(int(c) for c in df['random_label'].unique())

    return jsonify({
        "students": df.to_dict(orient='records'),
        "graph_image_url": url1,
        "graph_image2_url": url2,
        "unique_classes_allocated": unique_allocated_classes,
        "unique_classes_random_allocated": unique_random_allocated_classes
    })


""" @app.route('/interactive_graph')
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
    net = Network('600px', '50%', bgcolor="#222222", font_color="white")
    # Create a Pyvis network for random allocation
    net = Network('600px', '50%', bgcolor="#222222", font_color="white")

    # Disable physics (this keeps the nodes from floating around)
    net.show_buttons(filter_=['physics'])

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
    return redirect(url_for('static', filename='interactive_graph.html')) """


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


@app.route('/class_graph/<int:class_id>')
def class_graph(class_id):
    global G
    if G is None:
        return "Graph not available", 400

    response_data = {}

    # Check and create random_label graph
    nodes_random = [n for n, d in G.nodes(data=True) if d.get('random_label') == class_id]
    if nodes_random:
        subG_random = G.subgraph(nodes_random)
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        pos_random = nx.spring_layout(subG_random, seed=42)
        nx.draw(subG_random, pos_random, with_labels=True, node_size=60, edge_color='gray',
                node_color='skyblue', ax=ax1)
        image_path_random = f'static/images/class_{class_id}_random_graph.png'
        plt.savefig(image_path_random)
        plt.close(fig1)
        response_data["random_graph_url"] = url_for('static', filename=f'images/class_{class_id}_random_graph.png')
    else:
        response_data["random_graph_url"] = None

    # Check and create allocated_class graph
    nodes_allocated = [n for n, d in G.nodes(data=True) if d.get('allocated_class') == class_id]
    if nodes_allocated:
        subG_allocated = G.subgraph(nodes_allocated)
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        pos_allocated = nx.spring_layout(subG_allocated, seed=42)
        nx.draw(subG_allocated, pos_allocated, with_labels=True, node_size=60, edge_color='gray',
                node_color='lightgreen', ax=ax2)
        image_path_allocated = f'static/images/class_{class_id}_allocated_graph.png'
        plt.savefig(image_path_allocated)
        plt.close(fig2)
        response_data["allocated_graph_url"] = url_for('static', filename=f'images/class_{class_id}_allocated_graph.png')
    else:
        response_data["allocated_graph_url"] = None

    return jsonify(response_data)



current_model = None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
