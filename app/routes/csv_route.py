from flask import Blueprint, request, redirect, render_template, flash
from app.utils.csv_handler import upload_raw_csv, preprocess_raw_to_transformed
import os
from werkzeug.utils import secure_filename

main = Blueprint("main", __name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv"}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@main.route("/upload-csv", methods=["GET", "POST"])
def upload_csv():
    if request.method == "POST":
        file = request.files.get("file")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            try:
                upload_raw_csv(filepath)  # Insert raw data
                preprocess_raw_to_transformed()  # Auto-trigger transform
                flash("✅ CSV uploaded and processed successfully!", "success")
            except Exception as e:
                flash(f"❌ Error processing file: {e}", "danger")
            return redirect("/upload-csv")

    return render_template("upload_csv.html")


    
@main.route("/preprocess", methods=["POST"])
def preprocess():
    try:
        preprocess_raw_to_transformed()
        flash("✅ Preprocessing completed successfully.", "success")
    except Exception as e:
        flash(f"❌ Error during preprocessing: {e}", "danger")
    return redirect("/admin")