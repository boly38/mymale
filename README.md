# mymale
simple bird identification mvp based on trained tensorflow model

## Introduction
mymale: "My machine learning sample"

The goal of this project was to have an autonomous ready to play bird identification method
that rely on pre-trained model.

- "autonomous" meaning here: no need of notebook, and fixed dependencies versions.

## Run - identify a bird

Requirements :
- python setup : cf. [readme](./PYTHON_SETUP.md)
- data : cf [readme](data/README.md) : `./data/archive` must embed dataset + h5 model
  - `archive/EfficientNetB0-525-(224 X 224)- 98.97.h5` : a H5 pre-trained model to identify a bird (cf. § material).
  - `archive/test/` : test part of dataset to provision model label to show predicted class result

RUN !
- have a look at [`mymale.py`](./mymale.py)
```bash 
python mymale.py
```


## References and credits

### Original trigger
- [botEnSky wanted feature: identify a bird](https://github.com/boly38/botEnSky/issues/57)
- [Spidy20/Bird_Species_Classification_Streamlit](https://github.com/Spidy20/Bird_Species_Classification_Streamlit) and his [video](https://youtu.be/Ar6pCDWt2qs) that move me to this dataset 🙏

### Training about data science
- Kaggle website ["Learn" section](https://www.kaggle.com/learn) to train yourself with some simple notebook, dataset, learning model.

### Project material

📝 **Dataset & model used** : Gerry BIRDS 525 SPECIES- IMAGE CLASSIFICATION from Kaggle - [link](https://www.kaggle.com/datasets/gpiosenka/100-bird-species)

🛈 *Look at "code" tab, there is plenty samples jupyter notebook, especially the amazing ["What the bird"](https://www.kaggle.com/code/keloggs/what-the-bird) from
KUSHAGRA DWIVEDI*

NB: you could try to train a model from scratch, but it takes long time for this dataset ⌛ that's why we rely on pre-trained model (here `.h5` file) 


### Known issue
- [tensorflow - h5 wont load](https://discuss.tensorflow.org/t/teachable-machine-h5-wont-load/23005/10) - thanks Ajonn, Keti_Sulamanidze + [SO similar question](https://stackoverflow.com/questions/78187204/trying-to-export-teachable-machine-model-but-returning-error)