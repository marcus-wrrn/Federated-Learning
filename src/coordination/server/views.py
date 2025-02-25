from flask import request, jsonify, Blueprint, current_app
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
    return jsonify(asdict(round)), 200