import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import joblib
import os
import json
from datetime import datetime

# --- v2 Configuration ---
CONFIG = {
    "data_path": 'credit_risk_dataset.csv',
    "artifacts_dir": 'artifacts',
    "target_col": 'loan_status',
    "iv_threshold": 0.02,
    "scorecard_params": {
        "pdo": 40,
        "base_score": 700,
        "base_odds": 50
    },
    "lgbm_params": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": 42,
        "n_splits": 5  # For cross-validation
    }
}

# --- Utility Functions ---

def load_and_preprocess_data(path):
    """Loads data and handles missing values."""
    print("1. Loading and Preprocessing Data...")
    df = pd.read_csv(path)
    numeric_cols = df.select_dtypes(include=np.number).columns
    object_cols = df.select_dtypes(include='object').columns
    numeric_fills = {col: df[col].median() for col in numeric_cols}
    object_fills = {col: df[col].mode()[0] for col in object_cols}
    df.fillna(value={**numeric_fills, **object_fills}, inplace=True)
    return df

def calculate_woe_iv(df, feature, target):
    """Calculates Weight of Evidence and Information Value with smoothing."""
    df_woe_iv = df.groupby(feature)[target].agg(['count', 'sum']).reset_index()
    df_woe_iv.columns = [feature, 'total_count', 'bad_count']
    df_woe_iv['good_count'] = df_woe_iv['total_count'] - df_woe_iv['bad_count']
    df_woe_iv['good_count'] = df_woe_iv['good_count'].replace(0, 0.5)
    df_woe_iv['bad_count'] = df_woe_iv['bad_count'].replace(0, 0.5)
    total_good = df_woe_iv['good_count'].sum()
    total_bad = df_woe_iv['bad_count'].sum()
    df_woe_iv['dist_good'] = df_woe_iv['good_count'] / total_good
    df_woe_iv['dist_bad'] = df_woe_iv['bad_count'] / total_bad
    df_woe_iv['woe'] = np.log(df_woe_iv['dist_good'] / df_woe_iv['dist_bad'])
    df_woe_iv['iv'] = (df_woe_iv['dist_good'] - df_woe_iv['dist_bad']) * df_woe_iv['woe']
    iv = df_woe_iv['iv'].sum()
    woe_map = dict(zip(df_woe_iv[feature], df_woe_iv['woe']))
    return woe_map, iv

def create_bins_and_woe(df, target, iv_threshold):
    """Performs binning and WoE transformation."""
    print("2. Performing Feature Engineering and Selection...")
    manual_bins = {
        'person_age': [0, 23, 26, 30, 36, np.inf],
        'person_income': [0, 38000, 55000, 78000, 100000, np.inf],
        'person_emp_length': [0, 2, 5, 8, np.inf],
        'loan_amnt': [0, 7500, 12000, 16000, np.inf],
        'loan_int_rate': [0, 7.5, 11, 13.5, np.inf],
        'loan_percent_income': [0, 0.2, 0.4, 0.55, np.inf],
        'cb_person_cred_hist_length': [0, 2, 4, 8, np.inf]
    }
    binned_features = []
    for feature, bins in manual_bins.items():
        bin_feature_name = f"{feature}_binned"
        df[bin_feature_name] = pd.cut(df[feature], bins=bins, right=False, labels=False)
        binned_features.append(bin_feature_name)
    
    woe_maps, iv_scores = {}, {}
    features_to_consider = [col for col in df.columns if col not in list(manual_bins.keys()) + [target]]
    for feature in features_to_consider:
        woe_map, iv = calculate_woe_iv(df, feature, target)
        if iv > iv_threshold:
            woe_maps[feature] = woe_map
            iv_scores[feature] = iv

    woe_df = pd.DataFrame()
    for feature, woe_map in woe_maps.items():
        woe_df[feature] = df[feature].map(woe_map).fillna(0)
    
    return woe_df, woe_maps, iv_scores, manual_bins, binned_features

def train_scorecard_model(X, y, woe_maps, params):
    """Trains the logistic regression model and builds the scorecard."""
    print("\n3. Training Model A (Scorecard)...")
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X, y)
    
    factor = params['pdo'] / np.log(2)
    offset = params['base_score'] - (factor * np.log(params['base_odds']))
    
    scorecard = {'base_score': offset, 'feature_points': {}}
    for i, feature in enumerate(X.columns):
        points, coeff = {}, lr_model.coef_[0][i]
        for category_val, woe_val in woe_maps[feature].items():
            points[str(category_val)] = -factor * coeff * woe_val
        scorecard['feature_points'][feature] = points
    
    auc = roc_auc_score(y, lr_model.predict_proba(X)[:, 1])
    print(f"Scorecard (Logistic Regression) Training AUC: {auc:.4f}")
    return lr_model, scorecard, auc

def train_lgbm_model(X, y, params):
    """Trains the LightGBM model using stratified k-fold cross-validation."""
    print(f"\n4. Training Model B (LightGBM) with {params['n_splits']}-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=params['n_splits'], shuffle=True, random_state=params['random_state'])
    oof_preds = np.zeros(len(X))
    models, importances = [], pd.DataFrame(index=X.columns)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**{k: v for k, v in params.items() if k != 'n_splits'})
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(10, verbose=False)])
        
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        models.append(model)
        importances[f'fold_{fold+1}'] = model.feature_importances_
        
    cv_auc = roc_auc_score(y, oof_preds)
    print(f"\nLightGBM Cross-Validation AUC: {cv_auc:.4f}")
    
    final_lgbm = lgb.LGBMClassifier(**{k: v for k, v in params.items() if k != 'n_splits'})
    final_lgbm.fit(X, y)
    
    return final_lgbm, cv_auc, importances

def save_artifacts(artifacts, config):
    """Saves all model artifacts and reports."""
    print("\n5. Saving Artifacts...")
    for name, artifact in artifacts.items():
        joblib.dump(artifact, os.path.join(config['artifacts_dir'], f"{name}.joblib"))
        
    report = {
        "training_timestamp_utc": datetime.utcnow().isoformat(),
        "model_versions": {"scorecard": "v2.0", "lgbm": "v2.0"},
        "evaluation_metrics": {
            "scorecard_training_auc": artifacts.pop("scorecard_auc"),
            "lgbm_cv_auc": artifacts.pop("lgbm_cv_auc")
        },
        "iv_scores": {k: round(v, 4) for k, v in artifacts.pop("iv_scores").items()},
        "lgbm_feature_importance": artifacts.pop("lgbm_importances").mean(axis=1).sort_values(ascending=False).to_dict()
    }
    with open(os.path.join(config['artifacts_dir'], 'evaluation_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"Artifacts and evaluation report saved in '{config['artifacts_dir']}' directory.")


if __name__ == "__main__":
    os.makedirs(CONFIG['artifacts_dir'], exist_ok=True)
    df = load_and_preprocess_data(CONFIG['data_path'])
    woe_df, woe_maps, iv_scores, manual_bins, binned_features = create_bins_and_woe(df, CONFIG['target_col'], CONFIG['iv_threshold'])
    lr_model, scorecard, scorecard_auc = train_scorecard_model(woe_df, df[CONFIG['target_col']], woe_maps, CONFIG['scorecard_params'])
    X_lgbm = pd.get_dummies(df.drop(CONFIG['target_col'], axis=1).drop(columns=binned_features), drop_first=True)
    lgbm_model, lgbm_cv_auc, lgbm_importances = train_lgbm_model(X_lgbm, df[CONFIG['target_col']], CONFIG['lgbm_params'])
    artifacts_to_save = {
        "woe_maps": woe_maps, "scorecard": scorecard, "manual_bins": manual_bins,
        "lgbm_model": lgbm_model, "lgbm_columns": X_lgbm.columns, "iv_scores": iv_scores,
        "scorecard_auc": scorecard_auc, "lgbm_cv_auc": lgbm_cv_auc, "lgbm_importances": lgbm_importances
    }
    save_artifacts(artifacts_to_save, CONFIG)
