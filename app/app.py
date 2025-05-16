from flask import Flask, request, render_template, jsonify, redirect
import pandas as pd
import torch
from utils import process_csvfile, train_model_from_csv, GAT, optimize_class_allocation
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import io
from flask import send_file, url_for
import networkx as nx
import matplotlib.cm as cm
import os



app = Flask(__name__)
G = None  # Global variable to store the graph
final_df = None  # Global variable to store the final DataFrame
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/allocate', methods=['POST'])
def allocate_students():
    global G
    global final_df
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
        df['random_label'] = np.random.randint(0, num_classes, size=len(df))

    df = optimize_class_allocation(df, num_classes)
    final_df = df.copy()
    for i in range(len(df)):
        G.nodes[i]['student_id'] = df.loc[i, 'student_id']
        G.nodes[i]['random_label'] = df.loc[i, 'random_label']
        G.nodes[i]['allocated_class'] = df.loc[i, 'allocated_class']
        G.nodes[i]['optimal_class'] = df.loc[i, 'optimal_class']
        G.nodes[i]['gender_code'] = df.loc[i, 'gender_code']
        G.nodes[i]['bullying_experience_flag'] = df.loc[i, 'bullying_experience_flag']

    url1 = graph_image()
    url2 = graph_image2()
    unique_allocated_classes = sorted(int(c) for c in df['optimal_class'].unique() if pd.notna(c))
    unique_random_allocated_classes = sorted(int(c) for c in df['random_label'].unique())

    return jsonify({
        "students": df.to_dict(orient='records'),
        "graph_image_url": url1,
        "graph_image2_url": url2,
        "unique_classes_allocated": unique_allocated_classes,
        "unique_classes_random_allocated": unique_random_allocated_classes
    })

@app.route('/download_csv')
def download_csv():
    global final_df  # Assuming df contains random_label and optimal_class

    columns_to_drop = ['block', 'allocated_class']
    df_filtered = final_df.drop(columns=columns_to_drop, errors='ignore')

    output = io.StringIO()
    df_filtered.to_csv(output, index=False)
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()), 
                     mimetype='text/csv',
                     as_attachment=True,
                     download_name='student_allocations.csv')


def graph_image():
    global G
    if G is None:
        return jsonify({"error": "No graph available. Please upload and allocate first."}), 400

    # Get number of unique classes
    allocated_classes = [data['optimal_class'] for _, data in G.nodes(data=True)]
    unique_classes = sorted(set(allocated_classes))
    class_to_color = {cls: i for i, cls in enumerate(unique_classes)}

    # Use a colormap
    cmap = cm.get_cmap('tab10', len(unique_classes))
    color_map = [cmap(class_to_color[G.nodes[n]['optimal_class']]) for n in G.nodes]

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

        node_colors = []
        node_border_colors = []
        for n in subG_random.nodes():
            data = subG_random.nodes[n]
            
            # Gender color
            if data.get('gender_code') == 0:
                color = 'lightblue'
            elif data.get('gender_code') == 1:
                color = 'lightpink'
            else:
                color = 'gray'
            node_colors.append(color)

            # Border color: red = influencer, white = isolated, black = normal
            if data.get('is_influencer'):
                border = 'green'
            elif data.get('is_isolated_at_risk'):
                border = 'red'
            else:
                border = (0, 0, 0, 0) 
            node_border_colors.append(border)

        fig1, ax1 = plt.subplots(figsize=(8, 6))
        pos_random = nx.spring_layout(subG_random, seed=42)
        # Create legend patches
        legend_elements = []

        # Example: Colors for gender
        legend_elements.append(mpatches.Patch(color='skyblue', label='Male'))
        legend_elements.append(mpatches.Patch(color='pink', label='Female'))

        # Example: Borders for influencers
        legend_elements.append(mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                                            markersize=10, label='Influencer', markerfacecolor='white', markeredgewidth=2))
        legend_elements.append(mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                                            markersize=10, label='At risk of isolation', markerfacecolor='white', markeredgewidth=2)) 

        # Add the legend
        ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)


        nx.draw(subG_random, pos_random, with_labels=True, node_size=60, edge_color='gray',
                node_color=node_colors, edgecolors=node_border_colors,ax=ax1, linewidths=3)
        image_path_random = f'static/images/class_{class_id}_random_graph.png'
        plt.savefig(image_path_random)
        plt.close(fig1)
        response_data["random_graph_url"] = url_for('static', filename=f'images/class_{class_id}_random_graph.png')
    else:
        response_data["random_graph_url"] = None

    # Check and create allocated_class graph
    nodes_allocated = [n for n, d in G.nodes(data=True) if d.get('optimal_class') == class_id]
    if nodes_allocated:
        subG_allocated = G.subgraph(nodes_allocated)

        node_colors = []
        for n in subG_allocated.nodes():
            data = subG_allocated.nodes[n]
            
            # Gender color
            if data.get('gender_code') == 0:
                color = 'lightblue'
            elif data.get('gender_code') == 1:
                color = 'lightpink'
            else:
                color = 'gray'
            node_colors.append(color)

            # Border color: red = influencer, white = isolated, black = normal
            if data.get('is_influencer'):
                border = 'green'
            elif data.get('is_isolated_at_risk'):
                border = 'red'
            else:
                border = (0, 0, 0, 0) 
            node_border_colors.append(border)

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        pos_allocated = nx.spring_layout(subG_allocated, seed=42)

       # Create legend patches
        legend_elements = []

        # Example: Colors for gender
        legend_elements.append(mpatches.Patch(color='skyblue', label='Male'))
        legend_elements.append(mpatches.Patch(color='pink', label='Female'))

        # Example: Borders for influencers
        legend_elements.append(mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                                            markersize=10, label='Influencer', markerfacecolor='white', markeredgewidth=2)) 
        legend_elements.append(mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                                            markersize=10, label='At risk of isolation', markerfacecolor='white', markeredgewidth=2)) 


        # Add the legend
        ax2.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)

        nx.draw(subG_allocated, pos_allocated, with_labels=True, node_size=60, edge_color='gray',
                node_color=node_colors, edgecolors=node_border_colors, ax=ax2,linewidths=3)
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
