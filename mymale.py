import sys
import tensorflow as tf
import matplotlib.pyplot as plt  # For plotting images
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator  # For feeding images for training
from tensorflow.keras.applications.resnet50 import preprocess_input  # preprocessing function for resnet50


def ping():
    print(sys.version)


class Mymale:
    def __init__(self, train_dir_path, h5_model_path):
        self.train_dir = train_dir_path
        self.h5_model = h5_model_path
        print(f'train:{train_dir_path} - h5Model:{h5_model_path}')
        self.labels = None
        self.model = None

    def load_labels_from_train(self):
        data_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input)  # data generator instance w/ preprocessing function for resnet 50
        batch_size = 1024  # Number of images loaded as a batch
        train_gen = data_gen.flow_from_directory(  # Training split data generator
            self.train_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        # result # Found 84635 images belonging to 525 classes.

        self.labels = {value: key for key, value in train_gen.class_indices.items()}
        print(f'{len(self.labels.values())} labels loaded')

    def load_model(self):
        path_model = self.h5_model
        self.model = tf.keras.models.load_model(path_model, custom_objects={'F1_score': 'F1_score'})
        print('model loaded')

    def predict_bird(self, path_image):
        if self.labels is None:
            self.load_labels_from_train()
        if self.model is None:
            self.load_model()
        img = tf.keras.utils.load_img(path_image,
                                      target_size=(224, 224))  #Loding image from the path and resizing to input size
        img_arr = tf.keras.utils.img_to_array(img)
        x = preprocess_input(np.expand_dims(img_arr.copy(), axis=0))  #Applying required preprocessing
        prediction = self.model.predict(x)
        predicted_label = self.labels[np.argmax(prediction)]
        print(predicted_label)
        # jupyter notebook only : this shows bird image
        # plt.title(f"Prediction: {predicted_label}")
        # plt.imshow(img)
        # plt.axis("off")

ping()

# ############ setup
current_dir = os.getcwd()
dir_dataset = current_dir + "/data/archive"  # Path to dataset # from dataset full archive
dir_train = f"{dir_dataset}/train"  # path to training split
# dir_test=f"{dir_dataset}/test" #path to test split
# dir_valid=f"{dir_dataset}/valid" #path to validation split
h5_model = current_dir + '/data/archive/EfficientNetB0-525-(224 X 224)- 98.97.h5'  # trained model
# ############ setup end

mymale = Mymale(dir_train, h5_model)

image1 = current_dir + '/data/archive/test/ABYSSINIAN GROUND HORNBILL/1.jpg'
mymale.predict_bird(image1)

image2 = current_dir + '/data/archive/test/CAATINGA CACHOLOTE/2.jpg'
mymale.predict_bird(image2)

