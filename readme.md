# Classificazione immagini con Keras

Un esempio di creazione di un modello per la classificazione automatica di immagini con Tensorflow e Keras. Il modello utilizza il **transfer learning** a partire da un modello pre-addestrato **Xception**.

In questo caso il modello classifica lo stile architettonico di un edificio a partire da un'immagine, ma la logica può essere applicata a qualsiasi problema di classificazione multiclasse. 


## File
*   **notebook.ipynb**: contiene il notebook Jupyter con il processo di creazione del dataset, l'analisi esplorativa dei dati e l'addestramento di diversi modelli con parametri ottimizzati, fino alla scelta del modello più performante
*   **train.py**: contiene il codice per l'addestramento del modello finale
*   **app.py**: contiene un'API Flask per effettuare predizioni utilizzando il modello finale
*   **Pipfile** e **Pipfile.lock**: sono i file per la gestione delle dipendenze
*   La cartella **models** contiene il modello finale addestrato
*   La cartella **pictures** contiene alcune immagini casuali per testare il modello
*   **Dockerfile** contiene le istruzioni per costruire ed eseguire il container Docker

## Istruzioni per il Deployment
Creare ed eseguire il container Docker per distribuire il modello su un server:


1.  **Build dell'immagine Docker:**
    ```bash
    docker build -t image-classification .
    ```

3.  **Esecuzione del container**:
    ```bash
    docker run -p 5000:5000 image-classification
    ```

4.  **Test del modello**:

    Una volta che il container Docker è attivo e in esecuzione, si può testare il modello inviando un'immagine con curl:

    ```bash
    curl -X POST -F "image=@./pictures/random_building_1.jpeg" http://localhost:5000/predict
    ```

## Creazione del dataset di addestramento

Le immagini vanno organizzate in cartelle suddivise per classe. 

Ad esempio:

*   images/
    *   classe1/
        *   immagine1.jpg
        *   immagine2.png
        *   ...
    *   classe2/
        *   immagine3.jpeg
        *   immagine4.jpg
        *   ...
    *   classe3/
        *   immagine5.png
        *   immagine6.jpeg
        *    ...
    *   ...

Il codice di addestramento si occuperà poi di *splittare* il dataset in due, per utilizzare una parte delle immagini per il *training* e l'altra per la validazione. 

## Pulizia dei Dati

- Alcune immagini potrebbero dover essere convertite in formato JPEG utilizzando OpenCV per garantire la compatibilità e gestire diversi formati di file.


## Analisi Esplorativa dei Dati (EDA)

All'interno del file `notebook.ipnyb` c'è una sezione dedicata all'esplorazione del dataset:

-   **Galleria di Immagini:** È stata implementata una funzione per visualizzare una galleria di immagini con etichette di classe.
-   **Controllo del Formato File:** È stata utilizzata una funzione per controllare il formato di tutte le immagini.
-   **Analisi della Struttura della Directory:** L'analisi delle cartelle mostra:
    -   Il numero totale delle classi.
    -   Il numero totale di immagini nel dataset.
    -   La distribuzione delle immagini tra le diverse classi.
-   **Squilibrio delle Classi:** è possibile vedere se ci sono classi più rappresentate di altre.
-   **Dimensioni delle Immagini:** Sono state calcolate la larghezza e l'altezza medie, massime e minime delle immagini.
-   **Immagini Sovradimensionate:** Sono state identificate e ridimensionate le immagini significativamente più grandi della dimensione media delle immagini.

## Addestramento

-   **Caricamento dei Dati:** Keras `ImageDataGenerator` è stato utilizzato per caricare il dataset, dividerlo in set di addestramento e validazione (80%/20%).
-   **Transfer Learning:**
    -   Il modello pre-addestrato *Xception* è stato utilizzato come base, ma si può partire anche da un altro modello, come *EfficientNet* o *ResNet*.
    -   Il modello base è stato congelato per fungere da estrattore di *feature*.
    -   Sono stati aggiunti livelli personalizzati sopra il modello pre-addestrato per la classificazione.
-   **Ottimizzazione:**
    -   Il tasso di apprendimento è stato ottimizzato testando diversi tassi di apprendimento e tracciando l'accuratezza di validazione. È stato selezionato un tasso di apprendimento di 0,001.
    -   È stato implementato il *checkpointing* del modello per salvare il modello con le migliori prestazioni in base all'accuratezza di validazione durante l'addestramento.
-   **Miglioramento del Modello:**
    -   È stato aggiunto un livello interno completamente connesso al modello e sono state testate diverse dimensioni di questo livello.
    -   Sono stati introdotti livelli di *dropout* per la regolarizzazione ed è stato trovato il miglior valore di *dropout*.
-   **Aumento dei dati:**
    -   Diverse tecniche di *data augmetation*(ad esempio, rotazione, spostamenti, ridimensionamento, capovolgimenti) fornite da `ImageDataGenerator` sono state utilizzate per aumentare la dimensione del dataset e migliorare la generalizzazione.
    -   Le immagini aumentate sono state visualizzate per verificare la strategia di aumento.
-   **Addestramento con dataset aumentato:** Il modello è stato addestrato utilizzando il dataset aumentato.
-   **Utilizzo di un modello più grande:** È stata testata una dimensione di input più grande (299x299) per verificare se ci fosse un miglioramento dell'accuratezza.

## Predizione

-   Il modello migliore è stato caricato e valutato sul set di validazione.
-   È stata scritta una funzione personalizzata per testare e prevedere la categoria di un'immagine di input.

## Deployment

-   Il modello più performante viene salvato durante la fase di *checkpointing*
-  Una API realizzata con Flask consente di utilizzare il modello .keras originale (vedi il file `app.py`)
-   Il modello è stato distribuito su un server e il servizio dovrebbe essere accessibile per le richieste: è possibile testare il modello inviando una richiesta curl.

Sulla una macchina locale eseguire il comando:

```
curl -X POST -F "image=@<percorso-di-una-immagine-di-esempio.jpg>" http://217.160.226.158:5000/predict
```

Assicurati di sostituire `@<percorso-di-una-immagine-di-esempio.jpg>` con il percorso effettivo di un'immagine sulla tua macchina e dovresti ricevere una previsione.

Il risultato sarà un file json con la predizione della classe più probabile:


```
{
  "predicted_class": "beaux_arts",
  "probabilities": {
    "arabic": -2.452655076980591,
    "art_deco": 1.4610404968261719,
    "baroque": -3.5570831298828125,
    "beaux_arts": 3.9852452278137207,
    "brutalist": 0.10857202857732773,
    "byzantine": -2.243802547454834,
    "colonial": -1.1622830629348755,
    "gothic": -1.1947863101959229,
    "modernist": 0.32198435068130493,
    "palladian": -2.5474507808685303,
    "postmodern": 1.1707789897918701
  }
}

```