# ICON 25-26
#### Esame di ingegneria della conoscenza, UniBa, realizzato da: 

- Stefano Cici

- Antonio Bolsi

- Roberto Barracano


### ⚙ Setup iniziale dell'ambiente di lavoro:

0. Requisiti iniziali:

    Il progetto è stato creato sulla base del linguaggio di programmazione Python, quindi si raccomanda di installare la versione python         3.12.3, inoltre servirà anche un linguaggio di programmazione per la visione delle immagini, pertanto si raccomanda anche                 l'installazione di prolog 10.0.0-1

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
    Attraverso questo comando sarà possibile effettuare 6 operazioni prima che il menù sia disponibile: analisi statica dei valori conservati nel file CSV (contenente tutte le tipologie di diamante) e divisione di questi valori in 3 parti: high, medium e low; matrice di associazione (Cramèr's V), dove i valori più alti corrispondono ad una maggiore associazione; analisi distribuzioni dettagliate, dove viene fatto un confronto diretto di tutte le caratteristiche, con in allegato le distribuzioni delle singole caratteristiche; distribuzione prezzo per carati; distribuzione prezzo per taglio; Distribuzione prezzo per colore; distribuzione prezzo per chiarezza; matrice confusionale, costituita da 3 classi.
   Tutte queste informazioni verranno salvate in una sottocartella chiamata test_output, e successivamente verrà riportato nel terminale il menù principale, dove potranno essere svolte altre 5 operazioni differenti:
   1.   testare le previsioni dell'AI;
   2.   esplorare le soglie di valutazione;
   3.   addestrare il modello AI;
   4.   revisionare le informazioni del sistema;
   5.   uscita dal sistema

***

## Esecuzione del progetto
Aggiungere tutti i comandi che serviranno per le operazioni avviabili

