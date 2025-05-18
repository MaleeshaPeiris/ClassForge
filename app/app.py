from flask import Flask, request, render_template, jsonify, send_file, url_for
import pandas as pd
import torch
from utils import process_csvfile, train_model_from_csv, GAT, optimize_class_allocation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import io
import networkx as nx
import matplotlib.cm as cm
import os
from datetime import datetime

app = Flask(__name__)
G = None
G_old = None
final_df = None
current_model = None

@app.route('/')
def index():
    return render_template('index.html', active_page='allocator')

@app.route('/home')
def home():
    return render_template('home.html', active_page='home')

@app.route('/about')
def about():
    return render_template('about.html', active_page='about')

@app.route('/help')
def help():
    return render_template('help.html', active_page='help')

@app.context_processor
def inject_now():
    return {'now': datetime.utcnow}

from itertools import combinations

@app.route('/allocate', methods=['POST'])
def allocate_students():
    global G, final_df, current_model,G_old

    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    num_classes = int(request.form.get('num_classes', 3))
    academic_weight = int(request.form.get('academic_weight', 50))
    wellbeing_weight = int(request.form.get('wellbeing_weight', 50))

    train_dataset_df = pd.read_csv("student_dataset.csv")
    df = pd.read_csv(file)

    current_model = train_model_from_csv(train_dataset_df, num_classes, academic_weight, wellbeing_weight)
    data,G = process_csvfile(df, academic_weight, wellbeing_weight)

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

    # ✅ Step 4: Graph image URLs
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
    global final_df
    columns_to_drop = ['block', 'allocated_class']
    df_filtered = final_df.drop(columns=columns_to_drop, errors='ignore')

    output = io.StringIO()
    df_filtered.to_csv(output, index=False)
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name='student_allocations.csv')

def graph_image():
    global G
    if G is None:
        return jsonify({"error": "No graph available. Please upload and allocate first."}), 400

    allocated_classes = [data['optimal_class'] for _, data in G.nodes(data=True)]
    unique_classes = sorted(set(allocated_classes))
    class_to_color = {cls: i for i, cls in enumerate(unique_classes)}

    cmap = cm.get_cmap('tab10', len(unique_classes))
    color_map = [cmap(class_to_color[G.nodes[n]['optimal_class']]) for n in G.nodes]

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=45, node_color=color_map, edge_color='gray', ax=ax)

    image_path = os.path.join("static", "images", "graph_image.png")
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    plt.savefig(image_path)
    plt.close(fig)

    return url_for('static', filename='images/graph_image.png')

def graph_image2():
    global G
    if G is None:
        return jsonify({"error": "No graph available. Please upload and allocate first."}), 400

    allocated_classes = [data['random_label'] for _, data in G.nodes(data=True)]
    unique_classes = sorted(set(allocated_classes))
    class_to_color = {cls: i for i, cls in enumerate(unique_classes)}

    cmap = cm.get_cmap('tab10', len(unique_classes))
    color_map = [cmap(class_to_color[G.nodes[n]['random_label']]) for n in G.nodes]

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=45, node_color=color_map, edge_color='gray', ax=ax)

    image_path = os.path.join("static", "images", "graph2_image.png")
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    plt.savefig(image_path)
    plt.close(fig)

    return url_for('static', filename='images/graph2_image.png')

@app.route('/class_graph/<int:class_id>')
def class_graph(class_id):
    global G
    if G is None:
        return "Graph not available", 400

    response_data = {}

    # Random
    nodes_random = [n for n, d in G.nodes(data=True) if d.get('random_label') == class_id]
    if nodes_random:
        subG_random = G.subgraph(nodes_random)
        node_colors = []
        node_border_colors = []
        for n in subG_random.nodes():
            data = subG_random.nodes[n]
            color = 'lightblue' if data.get('gender_code') == 0 else 'lightpink' if data.get('gender_code') == 1 else 'gray'
            node_colors.append(color)
            node_border_colors.append('red' if data.get('is_influencer') else (0, 0, 0, 0))

        fig1, ax1 = plt.subplots(figsize=(8, 6))
        pos_random = nx.spring_layout(subG_random, seed=42)
        legend_elements = [
            mpatches.Patch(color='skyblue', label='Male'),
            mpatches.Patch(color='pink', label='Female'),
            mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=10, label='Influencer', markerfacecolor='white', markeredgewidth=2)
        ]
        ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
        nx.draw(subG_random, pos_random, with_labels=True, node_size=60, edge_color='gray',
                node_color=node_colors, edgecolors=node_border_colors, ax=ax1, linewidths=3)

        image_path_random = f'static/images/class_{class_id}_random_graph.png'
        plt.savefig(image_path_random)
        plt.close(fig1)
        response_data["random_graph_url"] = url_for('static', filename=f'images/class_{class_id}_random_graph.png')
    else:
        response_data["random_graph_url"] = None

    # Optimal
    nodes_allocated = [n for n, d in G.nodes(data=True) if d.get('optimal_class') == class_id]
    if nodes_allocated:
        subG_allocated = G.subgraph(nodes_allocated)
        node_colors = []
        node_border_colors = []
        for n in subG_allocated.nodes():
            data = subG_allocated.nodes[n]
            color = 'lightblue' if data.get('gender_code') == 0 else 'lightpink' if data.get('gender_code') == 1 else 'gray'
            node_colors.append(color)
            node_border_colors.append('green' if data.get('is_influencer') else (0, 0, 0, 0))

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        pos_allocated = nx.spring_layout(subG_allocated, seed=42)
        legend_elements = [
            mpatches.Patch(color='skyblue', label='Male'),
            mpatches.Patch(color='pink', label='Female'),
            mlines.Line2D([], [], color='green', marker='o', linestyle='None', markersize=10, label='Influencer', markerfacecolor='white', markeredgewidth=2)
        ]
        ax2.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
        nx.draw(subG_allocated, pos_allocated, with_labels=True, node_size=60, edge_color='gray',
                node_color=node_colors, edgecolors=node_border_colors, ax=ax2, linewidths=3)

        image_path_allocated = f'static/images/class_{class_id}_allocated_graph.png'
        plt.savefig(image_path_allocated)
        plt.close(fig2)
        response_data["allocated_graph_url"] = url_for('static', filename=f'images/class_{class_id}_allocated_graph.png')
    else:
        response_data["allocated_graph_url"] = None

    return jsonify(response_data)

@app.route('/class_counts')
def class_counts():
    global final_df
    if final_df is None:
        return jsonify({"error": "No allocation available"}), 400

    optimal_counts = final_df['optimal_class'].value_counts().sort_index().to_dict()
    random_counts = final_df['random_label'].value_counts().sort_index().to_dict()

    return jsonify({
        "optimal": optimal_counts,
        "random": random_counts
    })

@app.route('/class_students/<int:class_id>')
def class_students(class_id):
    global final_df
    if final_df is None:
        return jsonify({"error": "No data"}), 400

    filtered = final_df[final_df['optimal_class'] == class_id]
    student_data = []
    for _, row in filtered.iterrows():
        student_data.append({
            "student_id": row.get("student_id", "N/A"),
            "optimal_class": row.get("optimal_class", "N/A"),
            "random_class": row.get("random_label", "N/A"),
            "bully": "Yes" if row.get("is_bully", False) else "No",
            "gender": "Male" if row.get("gender_code") == 0 else "Female" if row.get("gender_code") == 1 else "Other"
        })

    return jsonify({"students": student_data})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
