# ClassForge GAT Allocation System

This is a modular Flask-based backend for **ClassForge**, an AI-powered classroom allocation system that uses a Graph Attention Network (GAT) pipeline and social network analysis to optimize student grouping. It supports CSV upload, preprocessing, GAT inference, and Neo4j graph export.

---

## 🗂️ Project Structure

```
app/
├── app.py                 # Flask entry point
├── __init__.py           # App factory
│
├── models/               # SQLAlchemy models
│   └── model.py
│
├── routes/               # Flask route blueprints
│   ├── admin_route.py       # Admin UI actions
│   ├── csv_route.py         # CSV upload and preprocessing
│   ├── gat_route.py         # Trigger GAT pipeline
│   ├── neo4j_route.py       # Neo4j export functions
│   └── route.py             # Base routes
│
├── services/            # GAT pipeline core logic
│   └── gat_pipeline.py
│
├── utils/               # Utility functions
│   ├── csv_handler.py      # CSV parsing and validation
│   └── utils.py            # Shared helper functions
│
├── static/              # Static files (JS for frontend)
│   └── script.js
│
└── templates/           # HTML templates
    ├── index.html
    ├── upload_csv.html
    ├── admin.html
    └── graph_preview.html
```

---

## 🚀 Features

- 📥 CSV Upload + Preprocessing
- 🤖 GAT Inference on Student Data
- 🧠 Embedding + Attention Extraction
- 🕸️ Graph Export to Neo4j + D3
- 👨‍💼 Admin Interface for triggering tasks

---

## 🛠️ Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Flask App

```bash
python app/app.py
```

App will be available at: [http://localhost:5000](http://localhost:5000)

---

## 📡 Key API Routes

| Endpoint                  | Method | Description                            |
|--------------------------|--------|----------------------------------------|
| `/upload-csv`            | POST   | Upload raw student CSV                 |
| `/preprocess`            | GET    | Transform raw → clean data             |
| `/run-gat`               | GET    | Trigger GAT allocation pipeline        |
| `/export-neo4j`          | GET    | Push GAT results to Neo4j              |
| `/graph-preview`         | GET    | Visualize D3-based classroom network   |

---

## 🧠 Notes

- Neo4j must be running before triggering `/export-neo4j`.
- GAT pipeline outputs include embeddings, attention scores, and edge types.
- Use `graph_preview.html` to debug exported relationships visually.

---

## 🔮 To Do

- Add GAT config sliders to admin UI
- Add filters in D3 visualization
- Add logging and exception handling
- Convert `gat_pipeline.py` into service class pattern

---

## 👤 Author

Developed as part of the **ClassForge** project — a smart education initiative powered by AI.
