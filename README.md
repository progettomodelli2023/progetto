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
  I risultati sono discussi nel paper ma in repository sono presenti sia un csv ( [metrics.csv](https://github.com/progettomodelli2023/progetto/blob/main/metrics.csv) ) metriche di valutazione della bontà della segmentazione (come Dice e Jaccard) Seguono maggiori dettagli su come il notebook è stato organizzato e su come riprodurre i nostri risultati.
 
