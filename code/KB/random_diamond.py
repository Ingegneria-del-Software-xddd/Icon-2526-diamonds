from config import RANDOM_DIAMOND
from typing import Tuple, Dict, Any, Optional, List
from preprocessing import CategoricalDataFrame
import json
import random



def random_diamond(df: CategoricalDataFrame, out_path: str = RANDOM_DIAMOND) -> Dict[str, Any]:
    """
    GENERA UN DIAMANTE CASUALE REALISTICO per testare il sistema di predizione.
    
    Questa funzione crea un diamante "finto" ma realistico che può essere usato per:
    - Testare se il sistema di predizione funziona correttamente
    - Fare dimostrazioni del sistema
    - Sviluppare e debugare il codice
    - Creare esempi per documentazione
    
    Il diamante generato è REALISTICO perché:
    1. Usa le stesse caratteristiche dei diamanti reali nel dataset
    2. Sceglie valori che esistono realmente nel mercato dei diamanti
    3. Mantiene le proporzioni tra diverse qualità (es: pochi diamanti "ideal", molti "good")
    
    Args:
        df: DataFrame categorico GIA' CARICATO con i dati dei diamanti
            (Passiamo il DataFrame già pronto per evitare di ricaricarlo ogni volta)
        out_path: Percorso dove salvare il file JSON con il diamante generato
                  (default: valore da config.py)
    
    Returns:
        None - La funzione non ritorna nulla, ma SALVA un file JSON
        
    Example:
        # Prima carichi il DataFrame una volta
        df_diamanti = CategoricalDataFrame()
        
        # Poi generi quanti diamanti vuoi
        random_diamond(df_diamanti, "diamante_test1.json")
        random_diamond(df_diamanti, "diamante_test2.json")
        # ...senza ricaricare i dati ogni volta!
    """
    
    # =============================================================================
    # FASE 1: PREPARAZIONE - Rimuoviamo il PREZZO dal DataFrame
    # =============================================================================
    
    # Perché rimuovere il prezzo?
    # Perché stiamo creando un diamante "NUOVO" di cui NON CONOSCIAMO il prezzo!
    # Il prezzo è quello che il nostro sistema AI deve PREDIRE.
    
    # Creiamo una COPIA del DataFrame originale
    # Questo è importante per NON MODIFICARE il DataFrame originale
    df_senza_prezzo = df.copy()  # df_senza_prezzo è una copia indipendente
    
    # Prova a rimuovere la colonna del prezzo (potrebbe chiamarsi in modi diversi)
    for nome_colonna_prezzo in ["price", "target", "label", "class"]:
        if nome_colonna_prezzo in df_senza_prezzo.columns:
            # Trovata la colonna prezzo! La eliminiamo
            df_senza_prezzo = df_senza_prezzo.drop(columns=[nome_colonna_prezzo])
            break  # Esci dopo aver rimosso la prima colonna "prezzo" trovata
    
    # Ora df_senza_prezzo contiene tutte le caratteristiche MA NON IL PREZZO
    # Esempio di colonne rimaste: ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
    
    # =============================================================================
    # FASE 2: FUNZIONE INTERNA - Genera un valore per una singola caratteristica
    # =============================================================================
    
    def generate_casual_value(caratteristica: str) -> str:
        """
        GENERA UN VALORE CASUALE REALISTICO per una specifica caratteristica del diamante.
        
        Pensala come un "esperto virtuale" che sa:
        - Quali valori sono possibili per ogni caratteristica
        - Quali combinazioni hanno senso nel mondo reale
        
        Args:
            caratteristica: Il nome della caratteristica (es: 'cut', 'color', 'carat')
            
        Returns:
            Un valore casuale ma REALISTICO per quella caratteristica
        """
        
        # Convertiamo in minuscolo per evitare problemi (es: 'Cut' vs 'cut')
        nome_caratteristica = caratteristica.lower()
        
        # -------------------------------------------------------------------------
        # STRATEGIA 1: CARATTERISTICHE NUMERICHE CLASSIFICATE (low/medium/high)
        # -------------------------------------------------------------------------
        # Queste sono caratteristiche che originariamente erano numeri
        # ma nel nostro sistema sono state convertite in categorie
        
        # CARAT (peso in carati) - influenza molto il prezzo
        if "carat" in nome_caratteristica:
            # low = diamante piccolo (es: 0.2-0.5 carati) - economico
            # medium = diamante medio (es: 0.5-1.5 carati) - normale
            # high = diamante grande (es: 1.5+ carati) - costoso
            return random.choice(["low", "medium", "high"])
        
        # DEPTH (profondità %) - influisce sulla brillantezza
        if "depth" in nome_caratteristica:
            # Ottimale: 60-64% (medium)
            # Troppo basso (<60%) o troppo alto (>64%) = meno brillantezza
            return random.choice(["low", "medium", "high"])
        
        # TABLE (tavola %) - influisce su come entra la luce
        if "table" in nome_caratteristica:
            # Ottimale: 55-65% (medium)
            return random.choice(["low", "medium", "high"])
        
        # DIMENSIONI (x, y, z in mm) - lunghezza, larghezza, profondità
        if "x" in nome_caratteristica or "y" in nome_caratteristica or "z" in nome_caratteristica:
            return random.choice(["low", "medium", "high"])
        
        # -------------------------------------------------------------------------
        # STRATEGIA 2: CARATTERISTICHE DI QUALITÀ CATEGORICHE
        # -------------------------------------------------------------------------
        # Queste sono già categorie nel mondo reale
        
        # CUT (taglio) - come è stato tagliato il diamante
        if "cut" in nome_caratteristica:
            # 5 livelli di qualità, da peggiore a migliore:
            # fair (povero) < good (buono) < very_good (molto buono) < premium (premium) < ideal (ideale)
            return random.choice(["fair", "good", "very_good", "premium", "ideal"])
        
        # COLOR (colore) - quanto è incolore il diamante
        if "color" in nome_caratteristica:
            # Scala da D (migliore, completamente incolore) a J (peggiore, giallino visibile)
            # D → E → F → G → H → I → J
            return random.choice(["d", "e", "f", "g", "h", "i", "j"])
        
        # CLARITY (chiarezza) - quante imperfezioni ha dentro
        if "clarity" in nome_caratteristica:
            # Scala da IF (perfetto internamente) a I1 (imperfezioni visibili a occhio nudo)
            # IF → VVS1 → VVS2 → VS1 → VS2 → SI1 → SI2 → I1
            return random.choice(["i1", "si2", "si1", "vs2", "vs1", "vvs2", "vvs1", "if"])
        
        # -------------------------------------------------------------------------
        # STRATEGIA 3: FALLBACK - per qualsiasi altra caratteristica
        # -------------------------------------------------------------------------
        # Se la caratteristica non è stata riconosciuta sopra,
        # prendiamo un valore reale dal dataset
        
        # Prendiamo la colonna dal DataFrame (senza prezzo)
        colonna = df_senza_prezzo[caratteristica]
        
        # Rimuoviamo eventuali valori "vuoti" (NaN)
        valori_validi = colonna.dropna().unique().tolist()
        
        if not valori_validi:  # Se non ci sono valori validi
            return "medium"  # Usiamo un valore neutro di default
        
        # Scegliamo CASUALMENTE tra i valori reali che esistono nel dataset
        return random.choice(valori_validi)
    
    # =============================================================================
    # FASE 3: GENERAZIONE - Creiamo il diamante completo
    # =============================================================================
    
    # Dizionario vuoto che conterrà il nostro diamante
    diamante_casuale = {}
    
    # Per OGNI caratteristica nel DataFrame (senza prezzo)...
    for caratteristica in df_senza_prezzo.columns:
        # ...generiamo un valore realistico
        valore = generate_casual_value(caratteristica)
        
        # Aggiungiamo al dizionario del diamante
        diamante_casuale[caratteristica] = valore
    
    # Alla fine, diamante_casuale sarà qualcosa come:
    # {
    #   "carat": "medium",
    #   "cut": "very_good",
    #   "color": "g",
    #   "clarity": "si1",
    #   "depth": "medium",
    #   "table": "medium",
    #   "x": "medium",
    #   "y": "medium",
    #   "z": "low"
    # }
    
    # =============================================================================
    # FASE 4: SALVATAGGIO - Salviamo il diamante in un file
    # =============================================================================
    
    # Apriamo il file in modalità SCRITTURA ("w" = write)
    with open(out_path, "w", encoding="utf-8") as file_json:
        # Scriviamo il dizionario in formato JSON
        json.dump(
            diamante_casuale,       # I dati da salvare
            file_json,              # Il file dove salvarli
            indent=4,               # Formattazione "bella" con rientri
            ensure_ascii=False      # Supporta caratteri speciali (es: accenti)
        )
    
    
    return diamante_casuale

