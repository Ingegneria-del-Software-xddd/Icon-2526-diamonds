from pathlib import Path as PathlibPath
from matplotlib.path import Path
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pyswip import Prolog
from config import PROLOG_FILE, CATEGORICAL_CSV, TARGET_COL,MODEL_PATH, CV_SPLITS 
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



class CategoricalDataFrame(pd.DataFrame):
    
    
    def __init__(self) -> None:
        super().__init__()
        self.prolog_to_categorical_dataframe()
        self.to_csv()
        self.train_model()
    
    
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


    def to_csv(self, path: str = CATEGORICAL_CSV) -> None:
        """
        Salva il DataFrame in un file CSV.
        """
            
        pd.DataFrame.to_csv(self, path, index=False)


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

            # MODIFICA 10: Titolo principale più piccolo
            #plt.suptitle("Matrice di Distribuzioni e Associazioni", fontsize=5, y=0.95)
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


    def plot_reliability_diagram(self, model_path: str = MODEL_PATH):
        """
        Genera un reliability plot per valutare la calibrazione delle probabilità del modello.
        
        Un reliability plot confronta le probabilità predette con le frequenze empiriche
        delle classi, mostrando quanto le probabilità siano ben calibrate.
        
        Args:
            model_path: Percorso del modello salvato
        """
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        
        # Carica il modello
        payload = joblib.load(model_path)
        model = payload["model"]
        le = payload.get("label_encoder")
        
        # Prepara i dati
        pre, selector, target, feats = self.build_preprocessor()
        X, y = self[feats], self[target]
        
        # Codifica target
        if le is None:
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            y_encoded = le.transform(y)
        
        # Ottieni probabilità predette
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)
            
            # Per multiclasse, possiamo valutare ogni classe separatamente
            n_classes = len(le.classes_)
            
            fig, axes = plt.subplots(1, n_classes, figsize=(5*n_classes, 5))
            if n_classes == 1:
                axes = [axes]
            
            for i, (cls_name, ax) in enumerate(zip(le.classes_, axes)):
                # Per la classe i, considera probabilità di appartenenza a questa classe
                prob_true, prob_pred = calibration_curve(
                    y_encoded == i, 
                    y_proba[:, i], 
                    n_bins=10,
                    strategy='uniform'
                )
                
                ax.plot(prob_pred, prob_true, marker='o', linewidth=1, label=f'Classe {cls_name}')
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfettamente calibrato')
                ax.set_xlabel('Probabilità predetta')
                ax.set_ylabel('Frazione osservata')
                ax.set_title(f'Reliability Plot - Classe {cls_name}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Calcola e stampa Brier score (metrica di calibrazione)
            from sklearn.metrics import brier_score_loss
            brier_scores = []
            for i in range(n_classes):
                brier = brier_score_loss(y_encoded == i, y_proba[:, i])
                brier_scores.append((le.classes_[i], brier))
                print(f"Brier score per classe {le.classes_[i]}: {brier:.4f}")
            
            return brier_scores
        else:
            print("Il modello non supporta predict_proba()")
            return None




    def train_model(self, model_path: str = MODEL_PATH, plot_reliability: bool = True) -> None:
        """
        Addestra il modello Random Forest con calibratore SENZA valutazioni.
        Le valutazioni vanno fatte con evaluate_model_performance() separatamente.
        """
        # 1. Preparazione dati e preprocessing
        pre, selector, target, feats = self.build_preprocessor()
        X, y = self[feats], self[target]
        
        # 2. Encoding della variabile target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)  # low=0, medium=1, high=2
        class_names = le.classes_
        
        # 3. Split train-test (solo per addestramento, il test sarà usato dopo)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
        )
        
        # 4. Costruzione della pipeline
        clf = RandomForestClassifier(
            n_estimators=300,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        )
        
        pipe = Pipeline([("pre", pre), ("sel", selector), ("clf", clf)])
        
        # 5. Calibrazione delle probabilità
        method: str = "sigmoid"  # isotonic necessita più dati
        cal = CalibratedClassifierCV(estimator=pipe, method=method, cv=3)
        
        # 6. Addestramento del modello (SOLO questo!)
        print("Addestramento del modello in corso...")
        cal.fit(X_train, y_train)
        print(f"✓ Modello addestrato con calibrazione ({method})")
        
        # 7. Preparazione del payload per il salvataggio
        payload = {
            "model": cal,
            "thresholds": {
                "decision_strategy": "argmax",
                "classes": class_names.tolist()
            },
            "features": feats,
            "calibrated": True,
            "calibration": {"method": method},
            "label_encoder": le,
            "class_names": class_names.tolist(),
            "train_test_split": {
                "X_train_shape": X_train.shape,
                "X_test_shape": X_test.shape,
                "random_state": 42
            }
        }
        
        # 8. Salvataggio del modello
        joblib.dump(payload, model_path)
        print(f"✓ Modello salvato in: {model_path}")
        
        if plot_reliability:
            self.plot_reliability_diagram(model_path=model_path)
                                         
    
    def plot_learning_curve_single_run(
        self,
        seed: int = 42,
        splits: int = 5,
        n_estimators: int = 300,
        sizes: int | list = 8,
        scoring: str = "f1_weighted",
        title: str | None = None
    ) -> None:
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
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Curva training score con intervallo di confidenza
        ax.plot(train_sizes, train_mean, marker="o", label="Training")
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
        
        # Curva validation score con intervallo di confidenza
        ax.plot(train_sizes, test_mean, marker="s", label="Cross-Validation")
        ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15)
        
        ax.set_xlabel("Training set size")
        ax.set_ylabel(scoring)
        
        # Titolo automatico o personalizzato
        if title is None:
            title = f"Learning Curve (seed={seed}, splits={splits}, n_estimators={n_estimators})"
        ax.set_title(title)
        
        ax.legend()
        plt.tight_layout()
        

        
        # MOSTRA GRAFICO:
        plt.show()
        
        
        
    '''        
        def plot_confusion_matrix(self, 
                            model_path: str = MODEL_PATH,
                            title: str = "Confusion Matrix") -> None:
        """
        Carica un modello salvato e mostra la matrice di confusione sul test set.
        
        Args:
            model_path: Percorso del file del modello salvato (default: MODEL_PATH)
            title: Titolo del grafico della matrice di confusione
        """
        # Carica il modello salvato
        try:
            payload = joblib.load(model_path)
            model = payload["model"]
            le = payload.get("label_encoder")
            class_names = payload.get("class_names")
            features = payload.get("features")
            
            print(f"Modello caricato da: {model_path}")
            print(f"Classi: {class_names}")
            print(f"Numero di feature: {len(features)}")
            
        except FileNotFoundError:
            print(f"Errore: File del modello non trovato in {model_path}")
            print("Esegui prima train_model() per addestrare e salvare il modello.")
            return
        except Exception as e:
            print(f"Errore nel caricamento del modello: {e}")
            return
        
        # Prepara i dati per la valutazione
        pre, selector, target, feats = self.build_preprocessor()
        
        # Assicurati che le feature siano le stesse usate durante l'addestramento
        if features is not None:
            X = self[features]
        else:
            X = self[feats]
        
        # Codifica la variabile target
        if le is None:
            le = LabelEncoder()
            y_encoded = le.fit_transform(self[target])
            class_names = le.classes_
        else:
            y_encoded = le.transform(self[target])
        
        # Divisione train-test (deve essere la stessa usata durante l'addestramento)
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
        )
        
        # Predizioni sul test set
        y_pred_test = model.predict(X_test)
        
        # Calcola la matrice di confusione
        cm = confusion_matrix(y_test, y_pred_test)
        
        # Visualizza la matrice di confusione
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Usa ConfusionMatrixDisplay per una visualizzazione standard
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names
        )
        
        disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)
        ax.set_title(title)
        
        # Aggiungi statistiche
        accuracy = accuracy_score(y_test, y_pred_test)
        ax.text(0.5, -0.15, f"Accuracy: {accuracy:.3f}", 
                transform=ax.transAxes, ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        # Stampa il report di classificazione
        print("\n=== Classification Report ===")
        print(classification_report(
            y_test, 
            y_pred_test, 
            target_names=class_names,
            digits=3
        ))        

        
    '''


    def evaluate_model_performance(self, model_path: str = MODEL_PATH, 
                                  plot_confusion_matrix: bool = True):
        """
        Valuta le performance di un modello salvato in modo indipendente.
        Valuta su TUTTO il dataset per coerenza e stabilità.
        
        Args:
            model_path: Percorso del file del modello salvato (default: MODEL_PATH)
            plot_confusion_matrix: Se True, mostra il grafico della matrice di confusione
        
        Returns:
            dict: Dizionario con tutte le metriche di performance
        """
        print(f"\n{'='*60}")
        print("VALUTAZIONE PERFORMANCE MODELLO".center(60))
        print('='*60)
        
        # 1. Carica il modello salvato
        try:
            payload = joblib.load(model_path)
            model = payload["model"]
            le = payload.get("label_encoder")
            features = payload.get("features")
            class_names = payload.get("class_names", ["low", "medium", "high"])
            
            print(f"✓ Modello caricato da: {model_path}")
            print(f"✓ Classi: {class_names}")
            print(f"✓ Numero di feature: {len(features) if features else 'N/A'}")
            
        except FileNotFoundError:
            print(f"✗ ERRORE: File del modello non trovato in {model_path}")
            raise
        except Exception as e:
            print(f"✗ ERRORE nel caricamento del modello: {e}")
            raise
        
        # 2. Prepara X e y dal DataFrame corrente
        if features is not None:
            X = self[features]
        else:
            X = self.drop(columns=['price'])
        
        if le is None:
            le = LabelEncoder()
            y_encoded = le.fit_transform(self['price'])
            class_names = le.classes_.tolist()
        else:
            y_encoded = le.transform(self['price'])
            # Assicurati che class_names sia una lista
            if not isinstance(class_names, list):
                class_names = list(class_names)
        
        y = y_encoded
        print(f"✓ Dimensioni dataset: {X.shape}")
        
        # CORREZIONE: Gestisci np.bincount in modo sicuro
        class_distribution = np.bincount(y)
        if hasattr(class_distribution, 'tolist'):
            class_distribution_list = class_distribution.tolist()
        else:
            class_distribution_list = list(class_distribution)
        
        print(f"✓ Distribuzione classi: {class_distribution_list}")
        print(f"✓ Valutazione su: TUTTO il dataset ({len(self)} campioni)")
        
        # 3. Cross-validation ROC-AUC
        metrics = {}
        
        if hasattr(model, 'predict_proba'):
            cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)
            
            try:
                cv_scores = cross_val_score(model, X, y, cv=cv, 
                                          scoring='roc_auc_ovo', n_jobs=-1)
                
                metrics['cv_roc_auc_mean'] = float(cv_scores.mean())
                metrics['cv_roc_auc_std'] = float(cv_scores.std())
                
                print(f"\n{' Cross-Validation ROC-AUC ':-^60}")
                print(f"Media (CV={CV_SPLITS}): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                
            except Exception as e:
                print(f"⚠ Cross-validation non disponibile: {e}")
                metrics['cv_roc_auc_mean'] = None
                metrics['cv_roc_auc_std'] = None
        
        # 4. Predizioni su tutto il dataset
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        
        # 5. Calcola metriche su tutto il dataset
        metrics['accuracy'] = float(accuracy_score(y, y_pred))
        metrics['f1_macro'] = float(f1_score(y, y_pred, average='macro', zero_division=0))
        metrics['f1_weighted'] = float(f1_score(y, y_pred, average='weighted', zero_division=0))
        
        if y_proba is not None:
            try:
                metrics['roc_auc_ovo'] = float(roc_auc_score(y, y_proba, multi_class='ovo', average='macro'))
                metrics['roc_auc_ovr'] = float(roc_auc_score(y, y_proba, multi_class='ovr', average='macro'))
            except:
                metrics['roc_auc_ovo'] = None
                metrics['roc_auc_ovr'] = None
        
        # 6. Stampare metriche complete
        print(f"\n{' Metriche Complete (su tutto il dataset) ':-^60}")
        print(f"Accuracy:           {metrics['accuracy']:.3f}")
        print(f"F1-score (macro):   {metrics['f1_macro']:.3f}")
        print(f"F1-score (weighted):{metrics['f1_weighted']:.3f}")
        
        if metrics.get('roc_auc_ovo') is not None:
            print(f"ROC-AUC OVO (macro): {metrics['roc_auc_ovo']:.3f}")
            print(f"ROC-AUC OVR (macro): {metrics['roc_auc_ovr']:.3f}")
        
        # 7. Report di classificazione
        print(f"\n{' Classification Report (su tutto il dataset) ':-^60}")
        y_original = le.inverse_transform(y)
        y_pred_original = le.inverse_transform(y_pred)
        
        print(classification_report(y_original, y_pred_original, 
                                    target_names=class_names, digits=3))
        
        # 8. MATRICE DI CONFUSIONE - Solo se richiesto
        if plot_confusion_matrix:
            print(f"\n{' Matrice di Confusione (su tutto il dataset) ':-^60}")
            
            # Calcola la matrice di confusione
            cm = confusion_matrix(y, y_pred)
            
            # Visualizza grafico
            fig, ax = plt.subplots(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=class_names
            )
            disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)
            ax.set_title(f"Matrice di Confusione - Tutto il Dataset (n={len(self)})")
            
            # Aggiungi statistiche
            accuracy = accuracy_score(y, y_pred)
            ax.text(0.5, -0.15, f"Accuracy: {accuracy:.3f} | Campioni: {len(self)}", 
                    transform=ax.transAxes, ha='center', fontsize=10)
            
            plt.tight_layout()
            plt.show()
            
            # 9. Mostra matrice di confusione in formato testuale
            print("\nMatrice di confusione (valori assoluti):")
            cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
            print(cm_df.to_string())
            
            # 10. Calcola e mostra accuratezza per classe
            print("\nAccuratezza per classe:")
            for i, class_name in enumerate(class_names):
                class_accuracy = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
                print(f"  {class_name}: {class_accuracy:.3f} ({cm[i, i]}/{cm[i].sum()})")
            
            # Salva la matrice nelle metriche
            metrics['confusion_matrix'] = cm.tolist()
            metrics['confusion_matrix_df'] = cm_df.to_dict()
        else:
            # Calcola comunque la matrice per le metriche, ma senza visualizzazione
            cm = confusion_matrix(y, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            print("\n⚠ Matrice di confusione non visualizzata (plot_confusion_matrix=False)")
        
        # 11. Aggiungi metadati alle metriche
        metrics['model_path'] = model_path
        metrics['dataset_size'] = len(self)
        metrics['n_features'] = X.shape[1]
        metrics['n_classes'] = len(class_names)
        metrics['class_names'] = class_names
        metrics['class_distribution'] = class_distribution_list
        metrics['evaluation_strategy'] = "full_dataset"
        metrics['plot_confusion_matrix'] = plot_confusion_matrix
        
        print(f"\n{' Valutazione completata ':-^60}")
        print(f"Dataset: {metrics['dataset_size']} campioni, {metrics['n_features']} feature")
        print(f"Classi: {len(class_names)} ({', '.join(class_names)})")
        print(f"Strategia: Valutazione su tutto il dataset")
        print(f"Matrice di confusione visualizzata: {'Sì' if plot_confusion_matrix else 'No'}")
        
        return metrics  
    







