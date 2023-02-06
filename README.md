# Semantic segmentation using Unet Encoder-Decoder for Brain Tumor MRI (BRATS2020)
Il progetto si occupa di fare semantic segmentation su diverse tipologie di risonanza magnetica al fine di classificare ogni pixel dell'immagine in una di tre label tra cui :
- WT, 
- TC,
- ET,
Abbiamo sviluppato e testato il codice qui presentato su Kaggle, una piattaforma che permette di collezzionare Dataset in cloud in modo tale da poter sfruttare agilmente le scansoni 3d (molto pesanti e complesse da maneggiare in locale).  
Il dataset considerato per questo task è [Brats2020](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data) che consiste in una collezione di scansioni multimodali in 3d del cervello di deiversi pazienti, in particolare :
- T1, native
- T1Gd, post-contrast T1-weighted
- T2, T2-weighted
- FLAIR, Fluid Attenuated Inversion Recovery T2
Ogni scansione ha dimensioni (240 x 240 x 155 )  
  
Il modello di rete neurale utilizzata abbiamo scelto un paradigma standard nel caso di semantic segmentation : [Unet](https://arxiv.org/abs/1505.04597)  
Si articola come una combinazione tra due modelli, un Encoder che estrapola i pattern rilevati da filtri convoluzionali in cascata ( in modo da poter interpolare informazioni a diverse granularità) e un Decoder che si pone l'obbiettivo di ricostruire l'immagine di input segmentata grazie all'embedding fornito dal Encoder.  

<p align="center"><img width="749" alt="unet" src="https://user-images.githubusercontent.com/124533848/217034682-8c0ef2b4-4b43-452a-bac1-ed249c3fb84f.png"></p>
  I risultati sono discussi nel paper ma in repository sono presenti sia un csv (metrics.csv) con metriche di valutazione della bontà della segmentazione (come Dice e Jaccard), sia una cartella di output del modello a fine del training. Seguono maggiori dettagli su come il notebook è stato organizzato e su come riprodurre i nostri risultati.
 
***

## Definizione classe BratsDataset per collezionare il dataset per training e validazione

E' stata definita la classe BratsDataser per organizzare il datapoint di input e di grund-truth in quanto il processo di training è supervised ( ovvero propagando una loss di errore sull'effettiva segmentazione disponibile dal dataset). In maggior dettaglio vengono letti da file di tipo .nii le diverse tipologie di scansioni relative a un particolare paziente e in seguito vengono costrite tre immagini di segmentation per ogni label di output. Su kaggle sono presenti dei csv utili per splittare i pazienti da considerare come datapoint di training e di testing, sfruttando questi abbiamo tutto l'occorrente per generare i DataLoader utili per il training e la validazione del nostro modello.

## Definizione funzioni di loss e modello Unet

<img align="left" src="https://user-images.githubusercontent.com/124533848/217044417-d9866ed1-9c7b-4bce-83eb-9f6f7c17b371.png" alt="dice" width="300"/>  Una tipica metrica che viene considerata in sematic segmentation è il Dice score, che corrisponde al doppio del rapporto tra l'intersezione e la somma delle superfici.
