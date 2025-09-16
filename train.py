# -*- coding: utf-8 -*-
"""
VGG11 from scratch pour CIFAR-10 (32x32), simple — sans régularisation, sans callbacks
"""

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Affiche si un GPU est détecté (TF choisit automatiquement)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU détecté :", gpus)
else:
    print("⚠️ Pas de GPU détecté — entraînement sur CPU.")

# Timer minimal
class timer:
    def __init__(self):
        self.start = None
        self.stop = None
    def tic(self):
        self.start = time.time()
    def toc(self):
        self.stop = time.time()
    def res(self):
        return None if self.start is None or self.stop is None else self.stop - self.start

# VGG11 simple (pas de BatchNorm, pas de dropout)
def build_vgg11(input_shape):
    model = models.Sequential(name="VGG11_CIFAR10_simple")
    model.add(layers.Input(shape=input_shape))
    
    
    model.add(layers.Conv2D(32, (3,3), padding='same', use_bias=True))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(32, (3,3), padding='same', use_bias=True))
    model.add(layers.Activation('relu'))
    model.add(layers.SpatialDropout2D(0.25))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(64, (3,3), padding='same', use_bias=True))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, (3,3), padding='same', use_bias=True))
    model.add(layers.Activation('relu'))
    model.add(layers.SpatialDropout2D(0.25))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(128, (3,3), padding='same', use_bias=True))
    model.add(layers.Activation('relu'))
    model.add(layers.SpatialDropout2D(0.25))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))    
    
    model.add(layers.Conv2D(128, (3,3), padding='same', use_bias=True))
    model.add(layers.Activation('relu'))
    model.add(layers.SpatialDropout2D(0.25))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10, activation='softmax'))

    return model

# Chargement et préparation des données CIFAR-10
class dataset:
    def __init__(self, nb_epochs=20, batch_size=64):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        x_train = x_train.astype('float32') / 255.0
        x_test  = x_test.astype('float32') / 255.0

        self.x_train = x_train
        self.y_train = y_train

        self.x_test = x_test
        self.y_test = y_test

        self.input_shape = self.x_train.shape[1:]
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size

        # one-hot
        self.y_train = tf.keras.utils.to_categorical(self.y_train, 10)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, 10)

        print("Train:", self.x_train.shape, self.y_train.shape)
        print("Test: ", self.x_test.shape, self.y_test.shape)
        print("Epochs:", self.nb_epochs, " Batch size:", self.batch_size)

# Entraînement simple
def train_model(data):
    print("➡️ Construction du modèle...")
    model = build_vgg11(data.input_shape)

    # Optimiseur simple
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    t = timer()
    t.tic()
    history = model.fit(
        data.x_train, data.y_train,
        validation_data=(data.x_test, data.y_test),
        epochs=data.nb_epochs,
        batch_size=data.batch_size,
        shuffle=True,
        verbose=2
    )
    t.toc()
    print(f"✅ Entraînement terminé en {t.res():.1f} s")
    return model, history

# Évaluation
def test_model(data, model):
    loss, acc = model.evaluate(data.x_test, data.y_test, verbose=2)
    print(f"Test loss: {loss:.4f}  Test accuracy: {acc:.4f}")

# (Facultatif) affichage des courbes
def plot_history(history):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train loss'); plt.plot(history.history['val_loss'], label='val loss')
    plt.legend(); plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='train acc'); plt.plot(history.history['val_accuracy'], label='val acc')
    plt.legend(); plt.title('Accuracy')
    plt.show()

# MAIN
if __name__ == '__main__':
    # Hyperparamètres simples
    NB_EPOCHS = 50
    BATCH_SIZE = 64  # si VRAM limitée -> réduire à 32 ou 16

    data = dataset(nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE)
    model, history = train_model(data)
    test_model(data, model)

    # Sauvegarde
    model.save("CIFAR10_VGG11_simple.h5")
    np.save("CIFAR10_xtest.npy", data.x_test)
    np.save("CIFAR10_ytest.npy", data.y_test)

