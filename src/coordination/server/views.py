from flask import request, jsonify, Blueprint, current_app, render_template
from server.database_orm import CoordinationDB
from dataclasses import asdict

bp = Blueprint("view", __name__, url_prefix="/view")

@bp.route('/models/<super_round>', methods=['GET'])
def models(super_round: int):
    with CoordinationDB(current_app.config["DATAPATH"]) as db:
        model_accs = db.get_model_accuracies_by_super_round(super_round)
    
    return jsonify(model_accs), 200
    
@bp.route('/current_round', methods=['GET'])
def current_round():
    with CoordinationDB(current_app.config["DATAPATH"]) as db:
        round = db.get_current_round()
        if round is None:
            return jsonify({"is_training": False}), 200

        clients = db.get_trained_clients(round.super_round_id, round.round_id)
        model_accs = db.get_model_accuracies_by_super_round(round.super_round_id)
        is_aggregating = db.is_aggregating()

    data = asdict(round)
    data["is_training"] = True
    data["client_count"] = clients
    data["aggregating"] = is_aggregating
    if model_accs is not None and len(model_accs) != 0:
        data["current_model"] = model_accs[-1]

    return jsonify(data), 200


@bp.route('/', methods=['GET'])
def home():
    return render_template('index.html'), 200