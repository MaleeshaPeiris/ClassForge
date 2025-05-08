from flask import Blueprint, render_template
main = Blueprint('main', __name__)


@main.route("/admin", methods=["GET"])
def admin_panel():
    return render_template("admin.html")