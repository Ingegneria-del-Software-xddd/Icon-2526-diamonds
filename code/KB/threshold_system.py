from typing import Tuple, Dict, Any, Optional, List  # Tipizzazione per chiarezza e controllo di errori
from enum import Enum  # Per enumerazioni tipizzate
import numpy as np  # Libreria per operazioni numeriche avanzate
import pandas as pd  # Libreria per manipolazione dati tabulari
#from preprocessing import df # Importo df solo per accellerare il debugging
from config import MINIKB_PATH, EXKB_PATH
import json
from pathlib import Path






class BeautyLevel(Enum):

    """Livelli apprezzamento da parte dell'audience
        per ogni caratteristica del diamante."""   
        
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'


def get_hierarchy_level(categorical_value: str) -> int:
    """
    Converte un valore categorico in un valore numerico gerarchico.
    
    Args:
        categorical_value: Valore categorico (es. 'low', 'ideal', 'd', 'vvs1')
    
    Returns:
        Intero che rappresenta la posizione gerarchica (più alto = migliore qualità)
    """
    
    # Converti in lowercase per sicurezza
    categorical_value = categorical_value.lower()
    
    # ===== VALORI GENERALI: low/medium/high =====
    # Per: carat, price, depth, table, x, y, z
    general_map = {
        'low': 1,
        'medium': 2,
        'high': 3
    }
    
    if categorical_value in general_map:
        return general_map[categorical_value]
    
    # ===== CUT =====
    # fair < good < very_good < premium < ideal
    cut_map = {
        'fair': 1,
        'good': 2,
        'very_good': 3,
        'premium': 4,
        'ideal': 5
    }
    
    if categorical_value in cut_map:
        return cut_map[categorical_value]
    
    # ===== COLOR =====
    # j < i < h < g < f < e < d (j peggiore, d migliore)
    color_map = {
        'j': 1,
        'i': 2,
        'h': 3,
        'g': 4,
        'f': 5,
        'e': 6,
        'd': 7
    }
    
    if categorical_value in color_map:
        return color_map[categorical_value]
    
    # ===== CLARITY =====
    # i1 < si2 < si1 < vs2 < vs1 < vvs2 < vvs1 < if
    clarity_map = {
        'i1': 1,
        'si2': 2,
        'si1': 3,
        'vs2': 4,
        'vs1': 5,
        'vvs2': 6,
        'vvs1': 7,
        'if': 8
    }
    
    if categorical_value in clarity_map:
        return clarity_map[categorical_value]
    
    # Valore non riconosciuto
    return 0



class Threshold: 
    ''' 
    Rappresenta una soglia numerica con un livello di apprezzamento associato.
    
    '''
    
    def __init__(self, 
                feature: str, 
                operator: str, 
                value: str, 
                level: BeautyLevel=BeautyLevel.MEDIUM, 
                description: str="") -> None:
        
        self.feature = feature                                      # Caratteristica del diamante (es. 'carat', 'depth', ecc.)
        self.operator = operator                                    # Operatore di confronto (es. '<', '>=', ecc.)
        self.value = value                                          # Valore categorico della soglia
        self.level = level                                          # Livello di apprezzamento associato
        self.description = description                              # descrizione soglia
        self.hierarchical_level = get_hierarchy_level(self.value)   # livello nella gerarchia
    

    def show_threshold(self):
        
        thr = [{
            
            "feature": self.feature,
            "operator": self.operator,
            "value": self.value,
            "level": self.level,
            "description": self.description
        }]
        
        return thr
    


# potrebbe rientrare nella classe threshold, pero' stica mo scoccia farlo entrare
'''

def fuzzy_evaluate(value: str, 
                   threshold_value: str, 
                   operator: str, 
                   margin: float = 0.5) -> float:
    """
    Valutazione "soft" del superamento soglia con logica fuzzy per valori categorici.
    
    Invece di un semplice superamento binario, restituisce un grado
    di appartenenza continuo [0, 1] che indica quanto il valore
    è desiderabile/indesiderabile.
    
    Args:
        value: Valore categorico da valutare (es. 'low', 'ideal', 'd')
        threshold_value: Valore di soglia categorico
        operator: Operatore di confronto ('>', '<', '>=', '<=', '==')
        margin: Margine per transizione smooth in unità gerarchiche (default 0.5)
    
    Returns:
        Grado di desiderabilità tra 0.0 (poco desiderabile) e 1.0 (molto desiderabile)
    """
    if value is None or threshold_value is None:
        return 0.0
    
    # Converti in valori gerarchici numerici
    value_level = get_hierarchy_level(value)
    threshold_level = get_hierarchy_level(threshold_value)
    
    # Per operatori di uguaglianza
    if operator == "==":
        if value_level == threshold_level:
            return 1.0  # Completamente uguale
        else:
            # Grado di vicinanza basato sulla distanza
            distance = abs(value_level - threshold_level)
            if distance <= margin:
                return 1.0 - (distance / margin)
            else:
                return 0.0
    
    # Per operatori di disuguaglianza
    if operator == ">=":
        # Valore deve essere >= soglia
        if value_level >= threshold_level:
            # Completamente soddisfatto
            return 1.0
        else:
            # Quanto manca?
            distance = threshold_level - value_level
            if distance <= margin:
                # Parzialmente soddisfatto
                return 1.0 - (distance / margin)
            else:
                # Troppo lontano
                return 0.0
    
    elif operator == "<=":
        # Valore deve essere <= soglia
        if value_level <= threshold_level:
            # Completamente soddisfatto
            return 1.0
        else:
            # Quanto supera?
            distance = value_level - threshold_level
            if distance <= margin:
                # Parzialmente soddisfatto
                return 1.0 - (distance / margin)
            else:
                # Troppo lontano
                return 0.0
    
    elif operator == ">":
        # Valore deve essere > soglia (strettamente maggiore)
        if value_level > threshold_level:
            # Completamente soddisfatto
            return 1.0
        elif value_level == threshold_level:
            # Esattamente sulla soglia - 50% soddisfatto
            return 0.5
        else:
            # Quanto manca?
            distance = threshold_level - value_level
            if distance <= margin:
                # Parzialmente soddisfatto
                return 0.5 * (1.0 - (distance / margin))
            else:
                # Troppo lontano
                return 0.0
    
    elif operator == "<":
        # Valore deve essere < soglia (strettamente minore)
        if value_level < threshold_level:
            # Completamente soddisfatto
            return 1.0
        elif value_level == threshold_level:
            # Esattamente sulla soglia - 50% soddisfatto
            return 0.5
        else:
            # Quanto supera?
            distance = value_level - threshold_level
            if distance <= margin:
                # Parzialmente soddisfatto
                return 0.5 * (1.0 - (distance / margin))
            else:
                # Troppo lontano
                return 0.0
    
    return 0.0

'''

class MiniKB:
    
    
    def __init__(self):
        
        self._store: Dict[int,Threshold] = {}
        self.position = 0
        self.populate_default_thresholds()
        
    
    def insert_threshold(self,threshold: Threshold) -> None:
        ''' Inserisce una soglia per una caratteristica specifica. '''
        
        self._store[self.position] = threshold
        self.position += 1
        
    
    def get_threshold(self, position: int) -> Optional[Threshold]:
        ''' Recupera la soglia per una caratteristica specifica. '''
        
        return self._store.get(position)    
    
    
    def query(self,
              feature: Optional[str] = None,
              operator: Optional[str] = None,
              value: Optional[str] = None,
              level: Optional[BeautyLevel] = None,
              description_like: Optional[str] = None
              ) -> pd.DataFrame:
        
        ''' Esegue una query sulle soglie memorizzate, 
            filtrando per attributi specifici. '''
            
            
        rows = []
        
        for position, thresh in self._store.items():
            if feature is not None and thresh.feature != feature:
                continue
            if operator is not None and thresh.operator != operator:
                continue
            if value is not None and thresh.value != value:
                continue
            if level is not None and thresh.level != level:
                continue
            if description_like is not None and (description_like.lower() not in thresh.description.lower()):
                continue

            rows.append({
                'feature': thresh.feature,
                'operator': thresh.operator,
                'value': thresh.value,
                'level': thresh.level,
                'description': thresh.description,             
                'dataset_column': f"{thresh.feature}_class"
            })
             
        return pd.DataFrame(rows, columns=['feature', 'operator', 'value', 'level', 'description', 'dataset_column'])


    def populate_default_thresholds(self) -> None:
        """
        Popola la KB con valori di default per i diamanti.
        Basati su regole standard del mercato dei diamanti.
        """
        # CARAT (caratura) - Per un buon affare: caratura non troppo alta
        self.insert_threshold(Threshold(
            feature="carat",
            operator="<=",
            value="medium",
            level=BeautyLevel.HIGH,
            description="Caratura non superiore a medium per buon rapporto qualità-prezzo"
        ))
        
        # CUT (taglio) - Almeno very_good per qualità visiva
        self.insert_threshold(Threshold(
            feature="cut",
            operator=">=",
            value="very_good",
            level=BeautyLevel.MEDIUM,
            description="Taglio almeno very_good per buona brillantezza"
        ))
        
        # COLOR (colore) - Almeno H per evitare tinte troppo visibili
        self.insert_threshold(Threshold(
            feature="color",
            operator=">=",
            value="h",
            level=BeautyLevel.MEDIUM,
            description="Colore almeno H (H, G, F, E, D accettabili)"
        ))
        
        # CLARITY (chiarezza) - Almeno SI1 per poche inclusioni visibili
        self.insert_threshold(Threshold(
            feature="clarity",
            operator=">=",
            value="si1",
            level=BeautyLevel.MEDIUM,
            description="Chiarezza almeno SI1 per poche inclusioni visibili ad occhio nudo"
        ))
        
        # PRICE (prezzo) - Non superiore a medium per convenienza
        self.insert_threshold(Threshold(
            feature="price",
            operator="<=",
            value="medium",
            level=BeautyLevel.HIGH,
            description="Prezzo non superiore a medium per essere considerato conveniente"
        ))
        
        # DEPTH (profondità) - Valore ottimale medium (tra 60-64%)
        self.insert_threshold(Threshold(
            feature="depth",
            operator="==",
            value="medium",
            level=BeautyLevel.MEDIUM,
            description="Profondità ottimale (60-64%) per massima brillantezza"
        ))
        
        # TABLE (tavola) - Valore ottimale medium (tra 55-65%)
        self.insert_threshold(Threshold(
            feature="table",
            operator="==",
            value="medium",
            level=BeautyLevel.MEDIUM,
            description="Tavola ottimale (55-65%) per proporzioni bilanciate"
        ))


    def fuzzy_beauty_score(self, diamond: Dict[str, Any]) -> float:
        """
        Calcola un punteggio di qualità continuo usando logica fuzzy.
        
        Args:
            self: MiniKB con le soglie
            diamond: Dati del diamante {feature: valore}
        
        Returns:
            Punteggio di qualità medio tra 0.0 (bassa qualità) e 1.0 (alta qualità)
        """
        
        
        def fuzzy_evaluate(value: str, 
                        threshold_value: str, 
                        operator: str, 
                        margin: float = 0.5) -> float:
            """
            Valutazione "soft" del superamento soglia con logica fuzzy per valori categorici.
            
            Invece di un semplice superamento binario, restituisce un grado
            di appartenenza continuo [0, 1] che indica quanto il valore
            è desiderabile/indesiderabile.
            
            Args:
                value: Valore categorico da valutare (es. 'low', 'ideal', 'd')
                threshold_value: Valore di soglia categorico
                operator: Operatore di confronto ('>', '<', '>=', '<=', '==')
                margin: Margine per transizione smooth in unità gerarchiche (default 0.5)
            
            Returns:
                Grado di desiderabilità tra 0.0 (poco desiderabile) e 1.0 (molto desiderabile)
            """
            if value is None or threshold_value is None:
                return 0.0
            
            # Converti in valori gerarchici numerici
            value_level = get_hierarchy_level(value)
            threshold_level = get_hierarchy_level(threshold_value)
            
            # Per operatori di uguaglianza
            if operator == "==":
                if value_level == threshold_level:
                    return 1.0  # Completamente uguale
                else:
                    # Grado di vicinanza basato sulla distanza
                    distance = abs(value_level - threshold_level)
                    if distance <= margin:
                        return 1.0 - (distance / margin)
                    else:
                        return 0.0
            
            # Per operatori di disuguaglianza
            if operator == ">=":
                # Valore deve essere >= soglia
                if value_level >= threshold_level:
                    # Completamente soddisfatto
                    return 1.0
                else:
                    # Quanto manca?
                    distance = threshold_level - value_level
                    if distance <= margin:
                        # Parzialmente soddisfatto
                        return 1.0 - (distance / margin)
                    else:
                        # Troppo lontano
                        return 0.0
            
            elif operator == "<=":
                # Valore deve essere <= soglia
                if value_level <= threshold_level:
                    # Completamente soddisfatto
                    return 1.0
                else:
                    # Quanto supera?
                    distance = value_level - threshold_level
                    if distance <= margin:
                        # Parzialmente soddisfatto
                        return 1.0 - (distance / margin)
                    else:
                        # Troppo lontano
                        return 0.0
            
            elif operator == ">":
                # Valore deve essere > soglia (strettamente maggiore)
                if value_level > threshold_level:
                    # Completamente soddisfatto
                    return 1.0
                elif value_level == threshold_level:
                    # Esattamente sulla soglia - 50% soddisfatto
                    return 0.5
                else:
                    # Quanto manca?
                    distance = threshold_level - value_level
                    if distance <= margin:
                        # Parzialmente soddisfatto
                        return 0.5 * (1.0 - (distance / margin))
                    else:
                        # Troppo lontano
                        return 0.0
            
            elif operator == "<":
                # Valore deve essere < soglia (strettamente minore)
                if value_level < threshold_level:
                    # Completamente soddisfatto
                    return 1.0
                elif value_level == threshold_level:
                    # Esattamente sulla soglia - 50% soddisfatto
                    return 0.5
                else:
                    # Quanto supera?
                    distance = value_level - threshold_level
                    if distance <= margin:
                        # Parzialmente soddisfatto
                        return 0.5 * (1.0 - (distance / margin))
                    else:
                        # Troppo lontano
                        return 0.0
            
            return 0.0
        

        scores: List[float] = []
        
        # Calcola il grado di qualità per ogni feature
        for position, thr in self._store.items():
            
            if thr.feature not in diamond:
                continue
            
            val = diamond[thr.feature]
            if val is None:
                continue
            
            # Usa fuzzy_evaluate per ogni caratteristica
            score = fuzzy_evaluate(str(val), thr.value, thr.operator)
            scores.append(score)
        
        # Restituisce la media dei gradi di qualità
        return float(np.mean(scores)) if scores else 0.0


    def save_to_json(self) -> None:
        """
        Salva la MiniKB con tutte le soglie memorizzate in formato JSON.
        
        Args:
            MINIKB_PATH: Percorso del file JSON dove salvare la KB
                        (es: 'kb/minikb.json', 'data/knowledge_base.json')
        
        Returns:
            None
        """
        # Converti la KB in un dizionario serializzabile
        kb_data = {
            "metadata": {
                "type": "MiniKB",
                "version": "1.0",
                "num_thresholds": len(self._store),
                "position_counter": self.position
            },
            "thresholds": []
        }
        
        # Aggiungi tutte le thresholds in ordine di posizione
        for position in sorted(self._store.keys()):
            thr = self._store[position]
            threshold_data = {
                "position": position,
                "feature": thr.feature,
                "operator": thr.operator,
                "value": thr.value,
                "level": thr.level.value,  # Usa .value per l'Enum
                "description": thr.description,
                "hierarchical_level": thr.hierarchical_level
            }
            kb_data["thresholds"].append(threshold_data)
        
        # Crea la directory se non esiste
        path = Path(MINIKB_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Salva in JSON
        with open(MINIKB_PATH, 'w', encoding='utf-8') as f:
            json.dump(kb_data, f, indent=2, ensure_ascii=False)


    def load_from_json(self) -> None:
        """
        Carica una MiniKB da un file JSON.
        
        Args:
            MINIKB_PATH: Percorso del file JSON da caricare
        
        Returns:
            None
        """
        # Leggi il file JSON
        with open(MINIKB_PATH, 'r', encoding='utf-8') as f:
            kb_data = json.load(f)
        
        # Reset della KB corrente
        self._store.clear()
        self.position = 0
        
        # Ricostruisci le thresholds
        for thr_data in kb_data.get("thresholds", []):
            # Ricostruisci l'Enum BeautyLevel
            level_value = thr_data["level"]
            beauty_level = BeautyLevel(level_value)  # Converti stringa in Enum
            
            # Crea la threshold
            threshold = Threshold(
                feature=thr_data["feature"],
                operator=thr_data["operator"],
                value=thr_data["value"],
                level=beauty_level,
                description=thr_data["description"]
            )
            
            # Inserisci nella posizione originale
            position = thr_data.get("position", self.position)
            self._store[position] = threshold
            
            # Aggiorna il contatore di posizione
            if position >= self.position:
                self.position = position + 1



class ExtendedKB(MiniKB):
   
    
    def __init__(self):
        super().__init__()
        self.composite_rules: List[Dict[str,Any]]=[]


    def add_composite_rule(self, 
                           name: str, 
                           conditions: List[Tuple[str,str,Any]],
                           beautyLevel: BeautyLevel):
        
        self.composite_rules.append({
            "name": name,
            "conditions": conditions,
            "BeautyLevel": beautyLevel
        })

    
    def save_to_json(self) -> None:
        """
        Salva l'ExtendedKB con thresholds e regole composite in JSON.
        Sovrascrive il metodo della classe base.
        """
        # Salva thresholds dalla classe base
        super().save_to_json()
        
        # Prepara le regole composite per il JSON
        serializable_rules = []
        for rule in self.composite_rules:
            serializable_rule = {
                "name": rule["name"],
                "conditions": rule["conditions"],
                "BeautyLevel": rule["BeautyLevel"].value  # Usa .value per l'Enum!
            }
            serializable_rules.append(serializable_rule)
        
        # Aggiungi le regole composite in un file separato
        composite_data = {
            "metadata": {
                "type": "ExtendedKB_CompositeRules",
                "version": "1.0",
                "num_rules": len(self.composite_rules)
            },
            "composite_rules": serializable_rules
        }
        
        # Salva regole composite
        path = Path(EXKB_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(EXKB_PATH, 'w', encoding='utf-8') as f:
            json.dump(composite_data, f, indent=2, ensure_ascii=False)
        

    def load_from_json(self) -> None:
        """
        Carica ExtendedKB da file JSON.
        Sovrascrive il metodo della classe base.
        """
        # Carica thresholds dalla classe base
        super().load_from_json()
        
        # Prova a caricare regole composite
        try:
            with open(EXKB_PATH, 'r', encoding='utf-8') as f:
                composite_data = json.load(f)
            
            # Ricostruisci le regole composite con l'Enum
            loaded_rules = []
            for rule_data in composite_data.get("composite_rules", []):
                # Ricostruisci BeautyLevel da stringa a Enum
                beauty_level_str = rule_data.get("BeautyLevel", "medium")
                beauty_level = BeautyLevel(beauty_level_str)
                
                rule = {
                    "name": rule_data["name"],
                    "conditions": rule_data["conditions"],
                    "BeautyLevel": beauty_level  # Enum, non stringa
                }
                loaded_rules.append(rule)
            
            self.composite_rules = loaded_rules
            
        except FileNotFoundError:
            print(f"Nessun file regole composite trovato: {EXKB_PATH}")
            self.composite_rules = []
        except json.JSONDecodeError as e:
            print(f"Errore nel parsing JSON: {e}")
            print("Il file JSON potrebbe essere corrotto o incompleto")
            self.composite_rules = []
        except ValueError as e:
            print(f"Errore nel caricamento BeautyLevel: {e}")
            print("Assicurati che i valori BeautyLevel siano 'low', 'medium' o 'high'")
            self.composite_rules = []    

  

