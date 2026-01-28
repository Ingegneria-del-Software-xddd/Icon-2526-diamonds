# ICON 25-26
#### Esame di ingegneria della conoscenza, UniBa, realizzato da: 

- Stefano Cici

- Antonio Bolsi

- Roberto Barracano


### ⚙ Setup iniziale dell'ambiente di lavoro:

0. Requisiti iniziali:
- python 3.12.3
- prolog 10.0.0-1

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

3. Installare dipendenze (in seguito sostituito con pip install -r requirements.txt)
    ```py
    pip install pandas numpy scikit-learn matplotlib seaborn scipy pyswip joblib rdflib
    ```

4. Avviare il programma
    ```py
    python kb\ui.py
    ```

***

## Esecuzione del progetto


<img align="center" src="immagini/Screenshot_menu_principale.png" width=400>

All’avvio, il sistema presenta un menù principale testuale che permette di guidare l’utente in base alle funzionalità disponibili. 
È possibile testare la previsione AI sui diamanti inserendo o generando dati e ottenendo stime di prezzo con probabilità e livello di confidenza. Il sistema consente inoltre di esplorare e gestire soglie di valutazione, applicando regole esperte sulla qualità dei diamanti.
Un modulo dedicato permette l’esportazione della conoscenza in formato RDF/Turtle, con supporto a query SPARQL e generazione di report semantici. L’utente inoltre può riaddestrare il modello AI, analizzare i dati in modo esplorativo e verificare le prestazioni del sistema di apprendimento.
L’esecuzione termina selezionando l’opzione di uscita dal menu.

