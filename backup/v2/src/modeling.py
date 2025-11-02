import pandas as pd
from src.config import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

def validate_model_inputs(df: pd.DataFrame, expected_cols: list, model_name: str = "model"):
    missing_cols = [col for col in expected_cols if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in expected_cols]

    if missing_cols:
        raise ValueError(f"[{model_name}] Données incomplètes : colonnes manquantes {missing_cols}")
    if len(extra_cols) > 50:  # arbitrairement 50 pour signaler du bruit
        print(f"[{model_name}] ⚠️ Attention : {len(extra_cols)} colonnes en trop dans les inputs")

    df_checked = df[expected_cols].copy()

    # Optionnel : vérif type numérique
    non_numeric_cols = df_checked.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric_cols:
        print(f"[{model_name}] ⚠️ Colonnes non-numériques dans les inputs : {non_numeric_cols}")

    return df_checked


def prepare_model_input(df: pd.DataFrame, target: str = "IS_WIN", model=None, drop_odds=True, drop_player_absent=True, verbose=True):
    """
    Prépare les données pour l'entraînement ou la prédiction.

    Args:
        df (pd.DataFrame): Données brutes contenant toutes les colonnes.
        target (str): Soit "IS_WIN" soit "POINT_DIFF".
        model: pipeline déjà entraîné (optionnel, pour vérifier les features exactes)
        drop_odds (bool): si True, on retire les colonnes de cotes.
        verbose (bool): Affiche les colonnes manquantes / en trop.

    Returns:
        pd.DataFrame: features prêtes à être utilisées.
    """

    if target == "IS_WIN":
        drop_cols = COLS_TO_DROP_TARGET_IS_WIN
    elif target == "POINT_DIFF":
        drop_cols = COLS_TO_DROP_TARGET_POINT_DIFF
    else:
        raise ValueError("Target non reconnue. Choisir 'IS_WIN' ou 'POINT_DIFF'")

    if drop_odds:
        drop_cols += COLS_ODDS

    if drop_player_absent:
        drop_cols += player_absent_input_cols

    drop_cols += COLS_MATCH_REAL + features_to_roll + top_player_features_to_roll
    features = [col for col in df.columns if col not in drop_cols + [target]]
    
    #drop nan
    df = df.dropna(subset=features)

    if model:
        expected = list(model.feature_names_in_)
        missing = [c for c in expected if c not in features]
        extra = [c for c in features if c not in expected]

        if verbose:
            if missing:
                print(f"❌ Colonnes manquantes : {missing}")
            if len(extra) > 1:
                print(f"⚠️ Trop de colonnes inutiles : {len(extra)}. \n colonnes : {extra}")
                

        features = expected

    return df[features].select_dtypes(include=["number"]).copy()


def get_optimal_model_params(model_name: str, use_gpu: bool = True, max_cpu_jobs: int = 3):
    """
    Retourne les meilleurs paramètres pour un modèle donné en fonction des ressources disponibles.

    Args:
        model_name (str): Nom du modèle ('xgb', 'lgbm', 'cat', 'rf', etc.)
        use_gpu (bool): Utilise le GPU si possible
        max_cpu_jobs (int): Limite du nombre de threads CPU à utiliser

    Returns:
        dict: Paramètres à passer au modèle
    """
    params = {}
    if model_name == "xgb":
        params.update({
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "eval_metric": "logloss",
            "random_state": 42,
            "use_label_encoder": False,
        })
        if use_gpu:
            params["tree_method"] = "hist"
            params["device"] = "cuda"
        else:
            params["n_jobs"] = max_cpu_jobs

    elif model_name == "lgbm":
        params.update({
            "n_estimators": 150,
            "num_leaves": 64,
            "random_state": 42,
        })
        if use_gpu:
            params["device"] = "gpu"
        else:
            params["n_jobs"] = max_cpu_jobs

    elif model_name == "cat":
        params.update({
            "n_estimators": 200,
            "learning_rate": 0.05,
            "depth": 6,
            #"rsm": 0.8,
            "verbose": 0,
            "random_state": 42,
        })
        if use_gpu:
            params["task_type"] = "GPU"
        else:
            params["task_type"] = "CPU"
            params["thread_count"] = max_cpu_jobs

    elif model_name == "rf":
        params.update({
            "n_estimators": 200,
            "random_state": 42,
            "n_jobs": max_cpu_jobs
        })

    return params




def get_classification_metrics_df(y_true, y_pred, y_proba, model_name="model"):
    """
    Retourne un DataFrame (1 ligne) avec toutes les métriques de classification principales pour un modèle.
    """
    from sklearn.metrics import confusion_matrix
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba),
    }
    # Confusion matrix à plat
    cm = confusion_matrix(y_true, y_pred)
    metrics["cm_00"] = cm[0, 0] if cm.shape == (2, 2) else None
    metrics["cm_01"] = cm[0, 1] if cm.shape == (2, 2) else None
    metrics["cm_10"] = cm[1, 0] if cm.shape == (2, 2) else None
    metrics["cm_11"] = cm[1, 1] if cm.shape == (2, 2) else None
    return pd.DataFrame([metrics])
    
def evaluate_model(model, X_train, y_train, X_test, y_test, name="Model"):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]
    
    print(f"========= {name} evaluation =========")
    
    #print(f"{name} - Classification report:\n", classification_report(y_test, preds))
    #print(f"{name} - ROC AUC: {roc_auc_score(y_test, probas):.4f}")
    
    #print_classification_metrics(y_test, preds, probas)

    metrics_df = get_classification_metrics_df(y_test, preds, probas, name)
    
    return model, probas, metrics_df

def plot_learning_curve(model, X, y, name="Model", scoring='roc_auc'):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=3, scoring=scoring)
    plt.plot(train_sizes, test_scores.mean(axis=1))
    plt.title(f'{name} Learning Curve')
    plt.xlabel('Training examples')
    plt.ylabel(scoring)
    plt.show()
    