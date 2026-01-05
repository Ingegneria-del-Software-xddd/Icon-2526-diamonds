from pathlib import Path as PathlibPath
from matplotlib.path import Path
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pyswip import Prolog
from config import PROLOG_FILE, CATEGORICAL_CSV, TARGET_COL, PREPROCESSED_CSV, MODEL_PATH, CV_SPLITS 
from scipy.stats import chi2_contingency
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Dict, Any, Optional, List
from datetime import datetime, timedelta
import json


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score        
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    roc_curve,
    precision_score,
    recall_score,
)





def metrics_at(y_true: np.ndarray, y_pred_proba: np.ndarray, thr: Optional[float] = None) -> dict:
    """
    Calcola metriche di valutazione per modelli di classificazione,
    supportando sia problemi multiclasse che binari.
    
    Args:
        y_true: Array NumPy con le etichette vere (ground truth)
                - Per multiclasse: valori interi 0, 1, 2, ...
                - Per binaria: valori 0 o 1
                
        y_pred_proba: Array NumPy con le probabilità predette
                      - Per multiclasse: forma (n_samples, n_classes)
                        probabilità per ciascuna classe
                      - Per binaria: forma (n_samples,)
                        probabilità della classe positiva
                        
        thr: Soglia di decisione (threshold)
             - Se None: usa strategia argmax (per multiclasse)
             - Se float: usa soglia per binarizzazione (per binaria)
    
    Returns:
        dict: Dizionario con 5 metriche di valutazione
    """
    
    # =========================================================================
    # FASE 1: ANALISI INIZIALE DEL PROBLEMA
    # =========================================================================
    
    # Conta il numero di classi uniche nelle etichette vere
    # Questo aiuta a capire se siamo in un contesto multiclasse (>2 classi)
    # o binario (2 classi: 0 e 1)
    n_classes = len(np.unique(y_true))
    
    # =========================================================================
    # FASE 2: GESTIONE DEGLI SCENARI POTENZIALMENTE PROBLEMATICI
    # =========================================================================
    
    # CONTROLLO 1: Verifica coerenza tra numero di classi e uso della soglia
    # Se abbiamo più di 2 classi ma viene specificata una soglia (thr),
    # potremmo essere in uno scenario logicamente problematico:
    # - Le soglie hanno senso principalmente in contesto binario
    # - Per multiclasse, la decisione tipica è via argmax
    if n_classes > 2 and thr is not None:
        # Avvertimento informativo per l'utente
        print(f"ATTENZIONE: y_true ha {n_classes} classi ma stai usando soglia={thr}")
        print("Considera di usare thr=None per decisione argmax")
    
    # =========================================================================
    # FASE 3: DECISIONE DELLE PREDIZIONI FINALI
    # =========================================================================
    
    if thr is None:
        # SCENARIO MULTICLASSE: Decisione tramite probabilità massima
        # ------------------------------------------------------------
        
        # CONTROLLO 2: Verifica dimensionalità dell'input
        # Per argmax, ci aspettiamo una matrice 2D:
        # - righe: campioni
        # - colonne: probabilità per ciascuna classe
        if y_pred_proba.ndim == 1:
            # Se riceviamo un array 1D invece di 2D, solleviamo un errore
            # perché non possiamo applicare argmax su una singola dimensione
            raise ValueError("Per thr=None serve y_pred_proba 2D (n_samples, n_classes)")
        
        # Applica argmax lungo l'asse delle colonne (axis=1)
        # Per ogni campione, seleziona l'indice della classe con probabilità più alta
        y_pred = np.argmax(y_pred_proba, axis=1)
        
    else:
        # SCENARIO BINARIO (o pseudo-binario): Decisione tramite soglia
        # --------------------------------------------------------------
        
        # CONTROLLO 3: Gestione array multidimensionali
        # Se riceviamo una matrice 2D invece di un array 1D,
        # dobbiamo estrarre le probabilità della classe positiva
        if y_pred_proba.ndim != 1:
            # Avvertimento informativo
            print("ATTENZIONE: Con thr specificato, usando solo prima colonna di y_pred_proba")
            
            # Strategia di estrazione:
            # - Se ci sono più colonne (>1): prendi la seconda colonna (indice 1)
            #   (tipicamente contiene le probabilità della classe positiva)
            # - Se c'è una sola colonna: prendi quella (indice 0)
            y_pred_proba = y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba[:, 0]
        
        # Applica la soglia: probabilità >= soglia → classe 1, altrimenti classe 0
        y_pred = (y_pred_proba >= thr).astype(int)
    
    # =========================================================================
    # FASE 4: CALCOLO DELLE METRICHE DI VALUTAZIONE
    # =========================================================================
    
    # Tutte le metriche sono calcolate con 'weighted' average per gestire
    # eventuali squilibri tra le classi, e con zero_division=0 per evitare
    # errori quando una classe non ha esempi nel set di valutazione.
    
    return {
        # F1-score pesato: media armonica di precision e recall,
        # pesata per il supporto di ciascuna classe
        "F1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        
        # F1-score macro: media semplice delle F1 delle singole classi
        "F1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        
        # Accuratezza: percentuale di predizioni corrette
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        
        # Precision pesata: capacità di non etichettare come positivi i negativi
        "Precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        
        # Recall pesato: capacità di trovare tutti i positivi
        "Recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


class CategoricalDataFrame(pd.DataFrame):
    
    def __init__(self) -> None:
        super().__init__()
        self.prolog_to_categorical_dataframe()
        self.eda()
        self.train_model()
        self.plot_learning_curve_single_run()
    
    
    def prolog_to_categorical_dataframe(self: pd.DataFrame) -> None:
        """
        Converte il dataset Prolog dei diamanti in un DataFrame pandas
        SOSTITUENDO i valori numerici con le classificazioni categoriali
        definite nelle regole del codice prolog.
        """
        prolog = Prolog()
        prolog.consult(PROLOG_FILE)
    
        # Trova tutti i diamanti
        risultati = list(prolog.query("prop(Diamond, carat, _)"))
        diamond_ids = list(set([ris["Diamond"] for ris in risultati]))
        diamond_ids.sort(key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
    
        # Colonne ORIGINALI ma con i valori numerici SOSTITUITI dalle classificazioni
        colonne_finali = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z', 'price']
    
        # Prepara la struttura dati solo con le colonne originali
        dati = {colonna: [] for colonna in colonne_finali}
    
        # Estrai i dati per ogni diamante
        for diamond_id in diamond_ids:
            # Per ogni colonna del dataset originale
            for colonna in colonne_finali:
                # Se è una colonna numerica che ha una classificazione categoriale
                if colonna in ['carat', 'depth', 'table', 'x', 'y', 'z', 'price']:
                    # Prendi la versione CLASSIFICATA invece del valore numerico
                    classe_colonna = f"{colonna}_class"
                    query = list(prolog.query(f"prop({diamond_id}, {classe_colonna}, Value)"))
                    if query:
                        dati[colonna].append(query[0]["Value"])
                    else:
                        dati[colonna].append(None)
                else:
                    # Per colonne già categoriali (cut, color, clarity, price), prendi il valore diretto
                    query = list(prolog.query(f"prop({diamond_id}, {colonna}, Value)"))
                    if query:
                        dati[colonna].append(query[0]["Value"])
                    else:
                        dati[colonna].append(None)
    
        # Crea il DataFrame
        df = pd.DataFrame(dati)
    
        for col in df.columns:
            self[col] = df[col]


    def get_target_column(self: pd.DataFrame) -> str:
    
        if TARGET_COL in self.columns:
            return TARGET_COL
        else:
            raise ValueError("Colonna target", TARGET_COL,"non trovata nel DataFrame.")


    def eda(self, grafici: bool = True) -> None:
        """
        Esegue un'analisi esplorativa completa del dataset categoriale dei diamanti
        mantenendo la stessa struttura dell'analisi per dati numerici.
    
        Args:
            self: DataFrame pandas con dati categoriali dei diamanti
            grafici: Flag booleano che controlla se generare o meno i grafici (default: False)
    
        Returns:
            None - La funzione opera principalmente tramite side effects (print e visualizzazioni)
        """
    
    # =============================================================================
    # SEZIONE 1: ANALISI STATISTICA DESCRITTIVA E QUALITÀ DEI DATI
    # =============================================================================
    
        print("\n=== ANALISI STATISTICA DESCRITTIVA ===")
    
    # Per dati categoriali, statistiche descrittive diverse
        stats_descrittive = pd.DataFrame({
            'Tipo': self.dtypes,
            'Valori Unici': self.nunique(),
            'Valori Non Nulli': self.count(),
           'Valori Nulli': self.isna().sum(),
            'Moda': self.mode().iloc[0] if not self.empty else None,
            'Freq Moda': [self[col].value_counts().iloc[0] if not self[col].empty else 0 for col in self.columns]
        })
    
        print(stats_descrittive)
    
        print("\n=== VALORI NULLI PER COLONNA ===")
        null_counts = self.isna().sum()
        if null_counts.sum() == 0:
            print("Nessun valore nullo trovato!")
        else:
            print(null_counts)

    # =============================================================================
    # SEZIONE 2: CONTROLLO PER LA GENERAZIONE DEI GRAFICI
    # =============================================================================
    
        if not grafici:
            print("\nAnalisi statistica completata. Grafici disattivati.")
            return

    # =============================================================================
    # SEZIONE 3: VISUALIZZAZIONE DELLA DISTRIBUZIONE DEL TARGET
    # =============================================================================
    
    # Identificazione automatica della colonna target (come nell'originale)
        target = 'price'  # Nel tuo caso sappiamo che è 'price'
    
        if target in self.columns:
            plt.figure(figsize=(10, 6))
            sns.countplot(x=target, data=self, order=self[target].value_counts().index)
            plt.title(f"Distribuzione Classe Target ({target})")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print(f"Colonna target '{target}' non trovata")

    # =============================================================================
    # SEZIONE 4: MATRICE DI CORRELAZIONE (ADATTATA PER DATI CATEGORIALI)
    # =============================================================================
    
        print("\n=== MATRICE DI ASSOCIAZIONE CATEGORIALE ===")
    
    # Per dati categoriali usiamo Cramér's V o heatmap di frequenze
        colonne_numeriche = []  # Nel tuo caso tutte le colonne sono categoriali
        colonne_categoriali = self.columns.tolist()
    
        if len(colonne_categoriali) > 1:
        # Creiamo una matrice di associazione usando i coefficienti di Cramér's V
            
            def cramers_v(x, y):
                confusion_matrix = pd.crosstab(x, y)
                chi2 = chi2_contingency(confusion_matrix)[0]
                n = confusion_matrix.sum().sum()
                phi2 = chi2 / n
                r, k = confusion_matrix.shape
                phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
                rcorr = r - ((r-1)**2)/(n-1)
                kcorr = k - ((k-1)**2)/(n-1)
                return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
        
        # Calcola matrice Cramér's V
            cramers_matrix = pd.DataFrame(np.zeros((len(colonne_categoriali), len(colonne_categoriali))),
                                        index=colonne_categoriali, columns=colonne_categoriali)
        
            for i, col1 in enumerate(colonne_categoriali):
                for j, col2 in enumerate(colonne_categoriali):
                    if i == j:
                        cramers_matrix.iloc[i, j] = 1.0
                    else:
                        try:
                            cramers_matrix.iloc[i, j] = cramers_v(self[col1], self[col2])
                        except:
                            cramers_matrix.iloc[i, j] = 0.0
        
        # Heatmap della matrice di associazione
            plt.figure(figsize=(12, 10))
            sns.heatmap(cramers_matrix, annot=True, cmap="coolwarm", center=0, 
                       vmin=0, vmax=1, fmt='.2f')
            plt.title("Matrice di Associazione (Cramér's V)")
            plt.tight_layout()
            plt.show()
        
            print("Matrice Cramér's V (valori più alti indicano associazione più forte):")
            print(cramers_matrix.round(3))

    # =============================================================================
    # SEZIONE 5: PAIRPLOT DELLE VARIABILI PRINCIPALI (ADATTATO)
    # =============================================================================
    
    
        variabili_principali = ['carat', 'cut', 'color', 'clarity', target]
        variabili_presenti = [col for col in variabili_principali if col in self.columns]

        if len(variabili_presenti) >= 2:
            # Creiamo una griglia di countplot invece del pairplot tradizionale
            n_vars = len(variabili_presenti)
            
            # MODIFICA 1: Rimpiccioliamo notevolmente la figura
            fig, axes = plt.subplots(n_vars, n_vars, figsize=(12, 12))
            
            # MODIFICA 2: Aumentiamo molto lo spazio tra i subplot
            plt.subplots_adjust(wspace=0.5, hspace=0.5)
            
            for i, var_row in enumerate(variabili_presenti):
                for j, var_col in enumerate(variabili_presenti):
                    ax = axes[i, j]
                    
                    if i == j:
                        # Diagonali: distribuzione singola variabile
                        counts = self[var_row].value_counts().sort_index()
                        ax.bar(range(len(counts)), counts.values, color='skyblue', alpha=0.7)
                        
                        # MODIFICA 3: Titolo più piccolo
                        ax.set_title(f'Distribuzione {var_row}', fontsize=9, pad=8)
                        ax.set_xticks(range(len(counts)))
                        
                        # MODIFICA 4: Etichette X più piccole e con rotazione più accentuata
                        ax.set_xticklabels(counts.index, rotation=60, ha='right', fontsize=7)
                        
                        # MODIFICA 5: Etichette Y più piccole
                        ax.tick_params(axis='y', labelsize=7)
                    
                    else:
                        # Non-diagonali: heatmap delle frequenze incrociate
                        cross_tab = pd.crosstab(self[var_row], self[var_col])
                        im = ax.imshow(cross_tab.values, cmap='YlOrRd', aspect='auto')
                        
                        # MODIFICA 6: Titolo più piccolo per heatmap
                        ax.set_title(f'{var_row} vs {var_col}', fontsize=8, pad=6)
                        ax.set_xticks(range(len(cross_tab.columns)))
                        
                        # MODIFICA 7: Etichette molto più piccole per heatmap
                        ax.set_xticklabels(cross_tab.columns, rotation=60, ha='right', fontsize=6)
                        ax.set_yticks(range(len(cross_tab.index)))
                        ax.set_yticklabels(cross_tab.index, fontsize=6)
                        
                        # MODIFICA 8: Annotazioni più piccole o rimosse per tabelle grandi
                        if cross_tab.shape[0] <= 4 and cross_tab.shape[1] <= 4:
                            for ii in range(len(cross_tab.index)):
                                for jj in range(len(cross_tab.columns)):
                                    ax.text(jj, ii, f'{cross_tab.iloc[ii, jj]}', 
                                        ha="center", va="center", color="black", fontsize=6)
                        # MODIFICA 9: Per tabelle più grandi, visualizza solo i valori alti
                        elif cross_tab.shape[0] <= 6 and cross_tab.shape[1] <= 6:
                            for ii in range(len(cross_tab.index)):
                                for jj in range(len(cross_tab.columns)):
                                    if cross_tab.iloc[ii, jj] != 0:  # Solo valori non zero
                                        ax.text(jj, ii, f'{cross_tab.iloc[ii, jj]}', 
                                            ha="center", va="center", color="black", fontsize=5)

            plt.tight_layout()
            plt.show()

    # =============================================================================
    # SEZIONE 6: ANALISI DISTRIBUZIONI DETTAGLIATE (NUOVA)
    # =============================================================================
    
        print("\n=== ANALISI DISTRIBUZIONI DETTAGLIATE ===")
    
        for colonna in self.columns:
            print(f"\n{colonna.upper()}:")
            conteggi = self[colonna].value_counts()
            for valore, count in conteggi.items():
                percentuale = (count / len(self)) * 100
                print(f"  {valore}: {count} diamanti ({percentuale:.1f}%)")

    # =============================================================================
    # SEZIONE 7: ANALISI RELAZIONE TARGET CON ALTRE VARIABILI
    # =============================================================================
    
        if target in self.columns:
            print(f"\n=== RELAZIONE CON TARGET ({target}) ===")
        
            variabili_predictive = [col for col in self.columns if col != target]
        
            for var in variabili_predictive[:4]:  # Analizza solo prime 4 variabili
                print(f"\nRelazione {var} → {target}:")
                cross_tab = pd.crosstab(self[var], self[target], normalize='index') * 100
                print(cross_tab.round(1))
            
            # Heatmap per le relazioni più importanti
                if var in ['carat', 'cut', 'color', 'clarity']:
                    plt.figure(figsize=(10, 6))
                    sns.heatmap(cross_tab, annot=True, fmt='.1f', cmap='Blues')
                    plt.title(f"Distribuzione {target} per {var} (%)")
                    plt.tight_layout()
                    plt.show()


    def build_preprocessor(self):
        """
        Costruisce un preprocessore per dati categorici.
        
        Args:
            self: CategoricalDataFrame
            
        Returns:
            tuple: (preprocessor, selector, target_column, feature_names)
        """
        
        # Usa la colonna target specificata o quella di default
        target_col = self.get_target_column()
        
        # Verifica che target_col esista
        if target_col not in self.columns:
            raise ValueError(f"Colonna target '{target_col}' non trovata")
        
        # Definisci colonne ordinali (con ordine naturale)
        ordinal_features = ['carat', 'price', 'depth', 'table', 'x', 'y', 'z']
        # Rimuovi target se è nelle ordinali
        ordinal_features = [c for c in ordinal_features if c != target_col]
        
        # Definisci colonne nominali (senza ordine naturale o con gerarchia complessa)
        nominal_features = ['cut', 'color', 'clarity']
        
        # Filtra solo quelle presenti nel dataframe
        feature_cols = [c for c in self.columns if c != target_col]
        ordinal_cols = [c for c in ordinal_features if c in feature_cols]
        nominal_cols = [c for c in nominal_features if c in feature_cols]
        other_cols = [c for c in feature_cols if c not in ordinal_cols + nominal_cols]
        
        # Pipeline per ordinali
        if ordinal_cols:
            ordinal_t = Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ])
        
        # Pipeline per nominali
        if nominal_cols:
            nominal_t = Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
        
        # Pipeline per altre
        if other_cols:
            other_t = Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
        
        # Combina transformers
        transformers = []
        if ordinal_cols:
            transformers.append(("ordinal", ordinal_t, ordinal_cols))
        if nominal_cols:
            transformers.append(("nominal", nominal_t, nominal_cols))
        if other_cols:
            transformers.append(("other", other_t, other_cols))
        
        preprocessor = ColumnTransformer(transformers)
        selector = SelectKBest(score_func=chi2, k="all")
        
        return preprocessor, selector, target_col, feature_cols   

    """
    def train_model(self):
        pre, selector, target, feats = self.build_preprocessor()
        X, y = self[feats], self[target]
        
        # ENCODING: Converti y in numerico per le metriche
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)  # low=0, medium=1, high=2
        class_names = le.classes_
        
        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
        X_fit, X_cal, y_fit, y_cal = train_test_split(X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42)    
        
        clf = RandomForestClassifier(n_estimators=300,n_jobs=-1,class_weight="balanced",random_state=42,)
        pipe = Pipeline([("pre", pre), ("sel", selector), ("clf", clf)])
        cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)
        
        # ROC-AUC per multiclasse
        cv_scores = cross_val_score(pipe, X_train_full, y_train_full, cv=cv, scoring='roc_auc_ovo', n_jobs=-1)
        print(f"ROC-AUC OVO media (CV={CV_SPLITS}): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        pipe.fit(X_fit, y_fit)
        method: str = "isotonic" if len(y_cal) >= 100 else "sigmoid"
        # cal = CalibratedClassifierCV(pipe, method=method, cv="prefit")  # type: ignore
        cal = CalibratedClassifierCV(estimator=pipe, method=method, cv=3)  # NEW SYNTAX
        cal.fit(X_cal, y_cal)
        print(f"Calibrazione probabilita: metodo = {method}")

        # Per multiclasse, predict_proba ritorna matrice [n_samples, n_classes]
        proba_cal = cal.predict_proba(X_cal)
        
        # Soglie per multiclasse - approccio divers, per multiclasse non ha senso una soglia singola
        print("Nota: Per classificazione multiclasse, le soglie sono per classe")
        
        y_pred_cal = cal.predict(X_cal)
        
        from sklearn.metrics import f1_score
        f1_macro = f1_score(y_cal, y_pred_cal, average='macro', zero_division=0)
        f1_weighted = f1_score(y_cal, y_pred_cal, average='weighted', zero_division=0)
        
        print(f"F1-score macro (validation): {f1_macro:.3f}")
        print(f"F1-score weighted (validation): {f1_weighted:.3f}")

        # Test set
        proba_test = cal.predict_proba(X_test)
        y_pred_test = cal.predict(X_test)
        acc = accuracy_score(y_test, y_pred_test)
        
        # ROC-AUC per multiclasse
        auc_ovo = roc_auc_score(y_test, proba_test, multi_class='ovo', average='macro')
        auc_ovr = roc_auc_score(y_test, proba_test, multi_class='ovr', average='macro')
        
        print("\n=== Performance su test (probabilita calibrate) ===")
        print(f"Accuracy: {acc:.3f}")
        print(f"ROC-AUC OVO (macro): {auc_ovo:.3f}")
        print(f"ROC-AUC OVR (macro): {auc_ovr:.3f}\n")
        
        # Report di classificazione con nomi originali
        y_test_original = le.inverse_transform(y_test)
        y_pred_test_original = le.inverse_transform(y_pred_test)
        
        print(classification_report(y_test_original, y_pred_test_original,  target_names=class_names))

        # Matrice di confusione
        cm = confusion_matrix(y_test, y_pred_test)
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot()
        plt.title(f"Confusion matrix (test) - {len(class_names)} classi")
        plt.show()

        # Per multiclasse, salva le soglie come dizionario per classe
        # Possiamo salvare la probabilità massima per ogni predizione
        best_thresholds = {
            "decision_strategy": "argmax",  # Per multiclasse usiamo argmax delle probabilità
            "classes": class_names.tolist(),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted)
        }
        
        payload = {
            "model": cal,                    
            "thresholds": best_thresholds,   
            "features": feats,               
            "calibrated": True,              
            "calibration": {"method": method},
            "label_encoder": le,  # Salva anche il label encoder
            "class_names": class_names.tolist()
        }

        joblib.dump(payload, MODEL_PATH)
        print(f"Modello calibrato e soglie salvati in {MODEL_PATH}")                
    """

     
    def train_model(self):
        pre, selector, target, feats = self.build_preprocessor()
        X, y = self[feats], self[target]
        
        # ENCODING: Converti y in numerico per le metriche
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)  # low=0, medium=1, high=2
        class_names = le.classes_
        
        # ✅ FIX: Single train/test split (no calibration split)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=300,n_jobs=-1,class_weight="balanced",random_state=42,)
        pipe = Pipeline([("pre", pre), ("sel", selector), ("clf", clf)])
        cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)
        
        # ROC-AUC per multiclasse
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc_ovo', n_jobs=-1)
        print(f"ROC-AUC OVO media (CV={CV_SPLITS}): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # ✅ FIX: Use CalibratedClassifierCV with CV (not prefit), this will do internal CV for calibration
        method: str = "sigmoid"  # isotonic needs more data
        cal = CalibratedClassifierCV(estimator=pipe,method=method, cv=3)  # Does 3-fold CV internally on training data
        
        
        cal.fit(X_train, y_train)
        print(f"Calibrazione probabilita: metodo = {method} (con CV=3)")

        # Validation metrics on training set (from CV)
        y_pred_train = cal.predict(X_train)
        f1_macro = f1_score(y_train, y_pred_train, average='macro', zero_division=0)
        f1_weighted = f1_score(y_train, y_pred_train, average='weighted', zero_division=0)
        print(f"F1-score macro (training): {f1_macro:.3f}")
        print(f"F1-score weighted (training): {f1_weighted:.3f}")

        # Test set evaluation
        proba_test = cal.predict_proba(X_test)
        y_pred_test = cal.predict(X_test)
        acc = accuracy_score(y_test, y_pred_test)

        # ROC-AUC per multiclasse
        auc_ovo = roc_auc_score(y_test, proba_test, multi_class='ovo', average='macro')
        auc_ovr = roc_auc_score(y_test, proba_test, multi_class='ovr', average='macro')
        print("\n=== Performance su test (probabilita calibrate) ===")
        print(f"Accuracy: {acc:.3f}")
        print(f"ROC-AUC OVO (macro): {auc_ovo:.3f}")
        print(f"ROC-AUC OVR (macro): {auc_ovr:.3f}\n")
        
        # Report di classificazione con nomi originali
        y_test_original = le.inverse_transform(y_test)
        y_pred_test_original = le.inverse_transform(y_pred_test)
        print(classification_report(y_test_original, y_pred_test_original, target_names=class_names))

        # Matrice di confusione
        cm = confusion_matrix(y_test, y_pred_test)
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot()
        plt.title(f"Confusion matrix (test) - {len(class_names)} classi")
        plt.show()

        # Per multiclasse, salva le soglie come dizionario per classe
        best_thresholds = {"decision_strategy": "argmax", "classes": class_names.tolist(), "f1_macro": float(f1_macro), "f1_weighted": float(f1_weighted)}
        payload = {"model": cal, "thresholds": best_thresholds,"features": feats, "calibrated": True, "calibration": {"method": method},"label_encoder": le,"class_names": class_names.tolist()}

        joblib.dump(payload, MODEL_PATH)
        print(f"Modello calibrato e soglie salvati in {MODEL_PATH}")
    

    def plot_learning_curve_single_run(
        self,
        seed: int = 42,
        splits: int = 5,
        n_estimators: int = 300,
        sizes: int | list = 8,
        scoring: str = "f1_weighted",
        out_png: str | PathlibPath | None = None,  
        out_csv: str | PathlibPath | None = None,
        title: str | None = None,
    ) -> tuple[PathlibPath, PathlibPath | None]:
        """
        Genera learning curve per analizzare comportamento algoritmo con diverse dimensioni training set.
        """
        import matplotlib.pyplot as plt
        
        # Preprocessing
        pre, selector, target, feats = self.build_preprocessor()
        
        # Aggiungi LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(self[target])

        X, y = self[feats], y_encoded
        
        # Modello Random Forest
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=-1,
            class_weight="balanced",
            random_state=seed,
        )
        
        # Pipeline completa
        pipe = Pipeline([("pre", pre), ("sel", selector), ("clf", clf)])
        cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

        # DEFINIZIONE DIMENSIONI TRAINING SET:
        if isinstance(sizes, int):
            train_sizes = np.linspace(0.1, 1.0, sizes)
        else:
            train_sizes = np.array(sizes, dtype=float)

        # CALCOLO LEARNING CURVE:
        train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(
            estimator=pipe,
            X=X,
            y=y,
            train_sizes=train_sizes,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0,
            return_times=True,
        )

        # STATISTICHE SUI SCORES:
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # CREAZIONE GRAFICO:
        plt.figure(figsize=(8, 5))
        
        # Curva training score con intervallo di confidenza
        plt.plot(train_sizes, train_mean, marker="o", label="Training")
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
        
        # Curva validation score con intervallo di confidenza
        plt.plot(train_sizes, test_mean, marker="s", label="Cross-Validation")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15)
        
        plt.xlabel("Training set size")
        plt.ylabel(scoring)
        
        # Titolo automatico o personalizzato
        if title is None:
            title = f"Learning Curve (seed={seed}, splits={splits}, n_estimators={n_estimators})"
        plt.title(title)
        
        plt.legend()
        
        # ✅ CORREZIONE: Gestione del caso out_png = None
        if out_png is None:
            # Crea nome file automatico
            out_png = f"test_output/learning_curve_seed{seed}_splits{splits}_ne{n_estimators}.png"
        
        # Converti in PathlibPath
        out_png_path = PathlibPath(out_png)
        out_png_path.parent.mkdir(parents=True, exist_ok=True)
        
        # SALVATAGGIO GRAFICO:
        plt.tight_layout()
        plt.savefig(out_png_path, dpi=150)
        plt.close()

        # SALVATAGGIO DATI NUMERICI (OPZIONALE):
        out_csv_path = None
        if out_csv is not None:
            out_csv_path = PathlibPath(out_csv)
            out_csv_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({
                "train_size": train_sizes,
                "train_mean": train_mean,
                "train_std": train_std,
                "test_mean": test_mean,
                "test_std": test_std,
                "scoring": scoring,
                "seed": seed,
                "splits": splits,
                "n_estimators": n_estimators,
                "fit_times_mean": np.mean(fit_times, axis=1),
                "score_times_mean": np.mean(score_times, axis=1),
            }).to_csv(out_csv_path, index=False)

        return out_png_path, out_csv_path
    
    
    
    
    












