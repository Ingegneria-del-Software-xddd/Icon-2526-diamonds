# ICON 25-26
#### Esame di ingegneria della conoscenza, UniBa, realizzato da: 

- Stefano Cici

- Antonio Bolsi

- Roberto Barracano


### ⚙ Setup iniziale dell'ambiente di lavoro:

0. Requisiti iniziali:

    Il progetto è stato creato sulla base del linguaggio di programmazione Python, quindi si raccomanda di installare la versione python 3.12.3, inoltre servirà anche un linguaggio di programmazione per la visione delle immagini, pertanto si raccomanda anche l'installazione di prolog 10.0.0-1

3. Clonare il repository eseguendo il seguente comando su terminale:  
    ```
    git clone https://github.com/Ingegneria-del-Software-xddd/test-icon
    ```

4. Creare e attivare un nuovo ambiente virtuale
    ```py
    py -3.11 -m venv venv
    ```
    ```
    venv\Scripts\activate
    ```

5. Installare dipendenze (in seguito sostituito con pip install -r requirements.txt)
    ```py
    pip install pandas numpy scikit-learn matplotlib seaborn scipy pyswip joblib
    ```

6. Avviare il programma
    ```py
    python kb\ui.py
    ```
   Il progetto si configura come un sistema di Business Intelligence e Machine Learning dedicato alla classificazione del valore commerciale dei diamanti. L'obiettivo principale è comprendere quali caratteristiche fisiche e qualitative influenzino il prezzo (suddiviso nelle classi high, medium, low) e costruire un modello predittivo affidabile.

### Punti Chiave dell'Analisi:
1. Analisi Correlativa: Utilizzando l'indice Cramér's V (Figure 2), il progetto identifica che il fattore più determinante per il prezzo è la caratura (carat, associazione 0.68), seguita dalle dimensioni fisiche (x, y, z).

2. Profilazione dei Dati: Attraverso matrici di distribuzione percentuale (Figure 4-7), il sistema evidenzia pattern specifici: ad esempio, il 100% dei diamanti con caratura "high" ricade nella fascia di prezzo "high", mentre la purezza (clarity) mostra un impatto più sfumato.

3. Modellazione Predittiva: È stato implementato un modello di classificazione con probabilità calibrate, capace di distinguere le fasce di prezzo con un'ottima accuratezza (80% sul test set) e un'area sotto la curva ROC media di 0.90 in validazione.

4. Monitoraggio Performance: Il progetto include una valutazione dettagliata tramite matrice di confusione (Figure 8) e report di classificazione, mostrando una precisione perfetta (100%) nell'identificare i diamanti di alto e basso valore, con qualche incertezza fisiologica solo sulla fascia media.
   
Tutte queste informazioni verranno salvate in una sottocartella chiamata test_output, e successivamente verrà riportato nel terminale il menù principale, dove potranno essere svolte altre 5 operazioni differenti:
   1.   testare le previsioni dell'AI;
   2.   esplorare le soglie di valutazione;
   3.   addestrare il modello AI;
   4.   revisionare le informazioni del sistema;
   5.   uscita dal sistema


<img align="right" src="Immagini/Screenshot_menu_principale" width=400>
***

## Esecuzione del progetto
Aggiungere tutti i comandi che serviranno per le operazioni avviabili

