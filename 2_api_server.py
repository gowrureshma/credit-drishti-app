from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
import logging

app = Flask(__name__)

# --- v2 Configuration ---
CONFIG = {
    "artifacts_dir": 'artifacts',
    "decision_thresholds": {
        "approve": 0.15,
        "manual_review": 0.35
    },
    "required_fields": {
        'person_age': int, 'person_income': int, 'person_home_ownership': str,
        'person_emp_length': int, 'loan_intent': str, 'loan_grade': str,
        'loan_amnt': int, 'loan_int_rate': float, 'loan_percent_income': float,
        'cb_person_default_on_file': str, 'cb_person_cred_hist_length': int
    }
}

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Artifacts ---
def load_artifacts(path):
    try:
        artifacts = {
            "woe_maps": joblib.load(os.path.join(path, 'woe_maps.joblib')),
            "scorecard": joblib.load(os.path.join(path, 'scorecard.joblib')),
            "lgbm_model": joblib.load(os.path.join(path, 'lgbm_model.joblib')),
            "lgbm_columns": joblib.load(os.path.join(path, 'lgbm_columns.joblib')),
            "manual_bins": joblib.load(os.path.join(path, 'manual_bins.joblib'))
        }
        logging.info("All artifacts loaded successfully.")
        return artifacts
    except FileNotFoundError as e:
        logging.error(f"Artifact loading failed: {e}. Please run the training script first.")
        return None

artifacts = load_artifacts(CONFIG['artifacts_dir'])

# --- Input Validation Function ---
def validate_input(data, schema):
    errors = []
    for field, expected_type in schema.items():
        if field not in data:
            errors.append(f"Missing required field: '{field}'")
        elif not isinstance(data[field], expected_type):
            if expected_type == float and isinstance(data[field], int):
                data[field] = float(data[field]) # Coerce int to float
                continue
            errors.append(f"Invalid type for field '{field}'. Expected {expected_type.__name__}, got {type(data[field]).__name__}")
    return errors

@app.route('/predict', methods=['POST'])
def predict():
    if not artifacts:
        return jsonify({'status': 'error', 'message': 'Server is not ready. Models not loaded.'}), 503

    try:
        data = request.get_json(force=True)
        logging.info(f"Received prediction request: {data}")

        validation_errors = validate_input(data, CONFIG['required_fields'])
        if validation_errors:
            logging.warning(f"Input validation failed: {validation_errors}")
            return jsonify({'status': 'error', 'message': 'Input validation failed', 'errors': validation_errors}), 400
        
        input_df = pd.DataFrame([data])
        
        # Scorecard Prediction
        score_input_df = input_df.copy()
        for feature, bins in artifacts['manual_bins'].items():
            if feature in score_input_df.columns:
                bin_feature_name = f"{feature}_binned"
                score_input_df[bin_feature_name] = pd.cut(score_input_df[feature], bins=bins, right=False, labels=False)
        
        total_score = artifacts['scorecard']['base_score']
        scorecard_breakdown = {'Base Score': round(artifacts['scorecard']['base_score'])}
        for feature, points_map in artifacts['scorecard']['feature_points'].items():
            if feature in score_input_df.columns:
                category = str(score_input_df[feature].iloc[0])
                points = points_map.get(category, 0)
                if round(points) != 0:
                    scorecard_breakdown[f'Points for {feature}={category}'] = round(points)
                total_score += points
        
        # LightGBM Prediction
        lgbm_input_df = pd.get_dummies(input_df).reindex(columns=artifacts['lgbm_columns'], fill_value=0)
        risk_probability = artifacts['lgbm_model'].predict_proba(lgbm_input_df)[0][1]
        
        # Tunable Decision Logic
        if risk_probability <= CONFIG['decision_thresholds']['approve']:
            decision = "Approved"
        elif risk_probability <= CONFIG['decision_thresholds']['manual_review']:
            decision = "Manual Review"
        else:
            decision = "Rejected"
            
        response = {
            'status': 'success', 'decision': decision,
            'risk_probability_percent': round(risk_probability * 100, 2),
            'scorecard_score': round(total_score),
            'scorecard_breakdown': scorecard_breakdown
        }
        logging.info(f"Prediction successful: {decision} with {risk_probability*100:.2f}% risk.")
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'An internal server error occurred.'}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=False)