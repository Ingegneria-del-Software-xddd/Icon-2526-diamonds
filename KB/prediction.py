from config import MODEL_PATH, SORTED_COLS, CATEGORICAL_CSV, RANDOM_DIAMOND
from typing import Tuple, Dict, Any, Optional, List
from preprocessing import CategoricalDataFrame
import numpy as np
import pandas as pd
import joblib
import json
import random


def load_payload(model_path = MODEL_PATH) -> Dict[str, Any]:
    
    obj = joblib.load(model_path)
    
    if isinstance(obj, dict) and "model" in obj:
        return obj
    else:
        
        return {
            "model": obj,  
            "thresholds": {"f1_weighted": 0.5, "youden": 0.5}, 
            "features": None  
        }
        

def ensure_columns(X_new: pd.DataFrame) -> pd.DataFrame:

    expected = SORTED_COLS
    for col in expected:
        if col not in X_new.columns:
            X_new[col] = np.nan

    return X_new[expected]


def predict_diamond(
    diamond: Dict[str, Any],
    model_path = MODEL_PATH,
    thr_mode: str = 'f1',
    thr_value: Optional[float] = None
):
    
    data = load_payload(model_path)
    model = data['model']
    thresholds = data.get("thresholds", {})
    expected = data.get("features")
    
    # Gestione threshold (MODIFICA per multiclasse)
    if "decision_strategy" in thresholds:
        # Caso MULTICLASSE: non usiamo soglie, usiamo argmax
        thr_mode = "argmax"
        threshold = None
    else:
        # Caso BINARIO: logica originale ma migliorata
        if thr_mode == "f1":
            threshold = float(thresholds.get("f1_weighted", 0.5))
        elif thr_mode == "youden":
            threshold = float(thresholds.get("youden", 0.5))
        elif thr_mode == "fixed":
            threshold = float(thr_value) if thr_value is not None else 0.5
        else:
            threshold = 0.5
            thr_mode = "fixed"
    
    X_new = pd.DataFrame([diamond])
    
    # Garantisci colonne attese
    if expected is not None:
        X_new = ensure_columns(X_new)
    
    # PREDIZIONE: distinguere binario vs multiclasse
    if hasattr(model, 'predict_proba'):
        proba_matrix = model.predict_proba(X_new)
        
        if threshold is None:  # MULTICLASSE
            # Argmax per multiclasse
            label = int(np.argmax(proba_matrix, axis=1)[0])
            proba = float(np.max(proba_matrix, axis=1)[0])
            predicted_class = data.get("class_names", ["low", "medium", "high"])[label]
            
            return predicted_class, proba, None, "argmax"
        else:  # BINARIO
            # Assumiamo colonna 1 = classe positiva
            proba = proba_matrix[:, 1][0] if proba_matrix.shape[1] > 1 else proba_matrix[:, 0][0]
            label = int(proba >= threshold)
            
            return label, proba, threshold, thr_mode
    else:
        # Modello senza predict_proba
        label = model.predict(X_new)[0]
        return label, None, threshold, thr_mode    
        








