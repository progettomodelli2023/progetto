# Semantic segmentation using Unet Encoder-Decoder for Brain Tumor MRI (BRATS2020)
Il task di questo progetto è semantic segmentation su diverse tipologie di risonanza magnetica al fine di classificare ogni pixel dell'immagine in una di tre label tra cui :
* WT, "whole tumor" rappresenta l'estensione completa del fenomeno patologico
* TC, "tumor core" circoscrive la massa tumorale (che solitamente viene considerata come parte da asportare)
* ET,  "enhancing tumor" lesioni di tipo Gadolinium-enhancing che riflettono le zone attive  
  
Abbiamo sviluppato e testato il codice qui presentato su Kaggle, una piattaforma che permette di collezzionare Dataset in cloud in modo tale da poter sfruttare agilmente le scansoni 3d (molto pesanti e complesse da maneggiare in locale).  
Il dataset considerato per questo task è [Brats2020](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data) che consiste in una collezione di scansioni multimodali in 3d del cervello di deiversi pazienti, in particolare :
* T1, native
* T1Gd, post-contrast T1-weighted
* T2, T2-weighted
* FLAIR, Fluid Attenuated Inversion Recovery T2
Ogni scansione ha dimensioni (240 x 240 x 155 )  
  
Il modello di rete neurale utilizzato rappresenta uno standard nel caso di semantic segmentation : [Unet](https://arxiv.org/abs/1505.04597)  
Si articola come una combinazione tra due modelli, un Encoder che estrapola i pattern rilevati da filtri convoluzionali in cascata ( in modo da poter interpolare informazioni a diverse granularità) e un Decoder che si pone l'obbiettivo di ricostruire l'immagine di input segmentata grazie all'embedding fornito dal Encoder.  

<p align="center"><img width="749" alt="unet" src="https://user-images.githubusercontent.com/124533848/217034682-8c0ef2b4-4b43-452a-bac1-ed249c3fb84f.png"></p>
  I risultati sono discussi nel paper ma in repository sono presenti sia un csv (metrics.csv) con metriche di valutazione della bontà della segmentazione (come Dice e Jaccard), sia una cartella di output del modello a fine del training. Seguono maggiori dettagli su come il notebook è stato organizzato e su come riprodurre i nostri risultati.
 

***
  
## Riproducibilità dei nostri esperimenti con visualizzazione segmentazione
In modo tale da osservare in modo semplice e veloce i nostri risultati abbiamo fornito in libreria:
* il modello di rete pretrained, alla fine del training abbiamo salvato in una cartella di checkpoint il modello in modo tale da condurre esperimenti sulle prestazioni
* porzioni del dataset di validation, in quanto un download totale avrebbe rallentato il processo e al tempo stesso complicato il processo di riproducibilità
* csv di validation in modo tale da creare il Dataloader
  
I punti fondamentali per riprodurre il nostro progetto consistono nell'esecuzione di pochi passaggi. Si inizia importando il nostro notebook su Colab con un runtime di tipo GPU, in seguito eseguire le sezioni iniziali di import e di definizione classe Brats, infine eseguire la sezione di test con visualizzazione che presenta i plot di input (4 tipologie di risonanza magnetica precedentemente spiegate), plot della ground truth (ovvero il risultato atteso della segmentazione per ogni tipologia di label come ET, TC e WT) e relativa segmentazione del nostro modello. Questi risultati sono anche velocemente consultabili dalla cartella output.  

## Riproducibilità dei nostri esperimenti con valutazione in termini di Dice e Jaccard

<img align="right" src="https://user-images.githubusercontent.com/124533848/217172590-89986712-364d-4c87-b36d-52397c933740.png" width="350" />  Abbiamo già discusso sulla metrica Dice e ora introdurremo una metrica detta Jaccard, tipica per stimare le prestazioni di un modello di segmentazione. Il Jaccard index è una misura di similitudine che può essere utilizzata in vari campi, tra cui la semantic segmentation. Si basa sulla comparazione tra due set, dove l'intersezione rappresenta gli elementi in comune e l'unione rappresenta l'insieme completo. E' calcolato come rapporto tra l'intersezione e l'unione. Questa misura può essere utilizzata per confrontare le prestazioni di un algoritmo di segmentazione, poiché misura la corrispondenza tra le segmentazioni generate dall'algoritmo e la segmentazione corretta.  

  
  I punti fondamentali per riprodurre il nostro progetto consistono nell'esecuzione di pochi passaggi. Si inizia importando il nostro notebook su Colab con un runtime di tipo GPU, in seguito eseguire le sezioni iniziali di import e di definizione classe Brats, infine eseguire la sezione di test del risultato atteso della segmentazione per ogni tipologia di label come ET, TC e WT in termini di Dice e Jaccard. Questi risultati sono anche velocemente consultabili dal csv (metrics.csv). 

***

## Definizione classe BratsDataset per collezionare il dataset per training e validazione

E' stata definita la classe BratsDataset per organizzare il datapoint di input e di ground-truth in quanto il processo di training è supervised ( ovvero propagando una loss di errore sull'effettiva segmentazione disponibile dal dataset). In maggior dettaglio vengono letti da file di tipo .nii le diverse tipologie di scansioni relative a un particolare paziente e in seguito vengono costruite tre immagini di segmentation per ogni label di output. Su kaggle sono presenti dei csv utili per splittare i pazienti da considerare come datapoint di training e di testing, sfruttando questi abbiamo tutto l'occorrente per generare i DataLoader utili per il training e la validazione del nostro modello.

## Definizione funzioni di loss e modello Unet

<img  src="https://user-images.githubusercontent.com/124533848/217044417-d9866ed1-9c7b-4bce-83eb-9f6f7c17b371.png" alt="dice" width="300"/><img src="https://user-images.githubusercontent.com/124533848/217057175-88c41e34-bf96-4a46-8751-7fb7bb9a9070.png" alt = "bce" height="100"/>

  Una tipica metrica che viene considerata in semantic segmentation è il Dice score, che corrisponde al doppio del rapporto tra l'intersezione e la somma delle superfici.  Invece la BCE sta per binary cross-entropy ed è una loss classica per i problemi di classificazione binaria. Quest'ultima metrica, per quanto possa sembrare slegata dal nostro task, è congeniale alla misurazione dell'errore sulla segmentazione della singola label in quanto la ground-truth è codificata come un'immagine di 0 e 1 e le prediction vengono discretizzate in base a una treshold.  
In seguito siamo passati alla definizione della struttura della rete neurale con una cascata di filtri convoluzionali e una cascata di filtri deconvoluzionali per generare quindi la famosa struttura a encoder-decoder.

## Classe di training 
Abbiamo definito una classe UnetExperiment in modo tale da stabilire in modo agile i parametri del training, i datapoints di training e validazione, i parametri strutturali della rete e alcuni parametri dell'optimizer.  
Quest'ultimo è fondamentale al processo di training in quanto stabilisce l'algoritmo di aggiornamento dei pesi della rete in relazione alla loss di predizione sui dati di training. Come optimizer abbiamo scelto Adam, che è una scelta classica in ambito deep learning. Maggiori dettagli implementativi sono descritti nel notebook.
