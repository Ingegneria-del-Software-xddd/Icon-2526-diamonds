# ICON 25-26

### Diamond Price Prediction & Knowledge Base Reasoning


#### Esame di ingegneria della conoscenza, UniBa, realizzato da: 

- [Stefano Cici](https://github.com/Stefano04Cici) - 796648;

- [Antonio Bolsi](https://github.com/Antob0906) - 759125;

- [Roberto Barracano](https://github.com/Hue-Jhan) - 799467;

***

### ⚙ Setup iniziale dell'ambiente di lavoro:

0. Requisiti iniziali:
    - Python 3.12.3;
    - [Swi prolog](https://www.swi-prolog.org) 10.0.0-1;

1. Clonare il repository eseguendo il seguente comando su terminale:  
    ```
    git clone https://github.com/Ingegneria-del-Software-xddd/test-icon
    ```

2. Creare e attivare un nuovo ambiente virtuale
    ```py
    py -3.11 -m venv venv
    ```
    ```
    venv\Scripts\activate
    ```

3. Installare dipendenze necessarie:
    ```py
    pip install pandas numpy scikit-learn matplotlib seaborn scipy pyswip joblib
    ```

4. Avviare il programma
    ```py
    cd code/
    ```
    ```py
    python kb/ui_rdf.py
    ```

***

### 📍 Guida all'utilizzo

AAAAAAAAAAAAAAAAAAAAAA MODIFICARE QUESTA PARTE

Attraverso questo comando sarà possibile effettuare 6 operazioni prima che il menù sia disponibile: analisi statica dei valori conservati nel file CSV (contenente tutte le tipologie di diamante) e divisione di questi valori in 3 parti: high, medium e low; matrice di associazione (Cramèr's V), dove i valori più alti corrispondono ad una maggiore associazione; analisi distribuzioni dettagliate, dove viene fatto un confronto diretto di tutte le caratteristiche, con in allegato le distribuzioni delle singole caratteristiche; distribuzione prezzo per carati; distribuzione prezzo per taglio; Distribuzione prezzo per colore; distribuzione prezzo per chiarezza; matrice confusionale, costituita da 3 classi.
Tutte queste informazioni verranno salvate in una sottocartella chiamata test_output, e successivamente verrà riportato nel terminale il menù principale, dove potranno essere svolte altre 5 operazioni differenti:
   1.   testare le previsioni dell'AI;
   2.   esplorare le soglie di valutazione;
   3.   addestrare il modello AI;
   4.   revisionare le informazioni del sistema;
   5.   uscita dal sistema
      
#### Opzioni Principali:

immagine


