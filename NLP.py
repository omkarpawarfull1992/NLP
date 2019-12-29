#Reference
#https://developers.google.com/machine-learning/guides/text-classification

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

#Display in new window
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

#import read Function
from ReadFile import load_tweet_weather_topic_classification_dataset 

#Train and Test Data
Train,Test=load_tweet_weather_topic_classification_dataset("C:\\Users\\opawar\\Desktop\\Python\\Projects\\Project 1 NLP")
X_train=pd.DataFrame(Train[0])
X_train.columns=['Tweet']
Y_train=pd.DataFrame(Train[1])
Y_train.columns=['Label']
X_test=pd.DataFrame(Test[0])
X_test.columns=['Tweet']
Y_test=pd.DataFrame(Test[1])
Y_test.columns=['Label']

#Train Sample Matrix
NumberOfSamples=X_train.size
NumberOfClasses=Y_train.Label.value_counts().size
NumberOfSamplePerClass=Y_train.Label.value_counts()

from ReadFile import get_num_words_per_sample
from ReadFile import plot_sample_length_distribution
NumberOfWordsPerSample=get_num_words_per_sample(Train[0])
plot_sample_length_distribution(Train[0])

#Deciding on which method to choose
Ratio=NumberOfSamples/NumberOfWordsPerSample

#Tokenization
#unigram+bigram approach
#Matrix of TF-IDF features
from Tokanization import ngram_vectorize
x_train,x_test=ngram_vectorize(Train[0],Train[1],Test[0])

#Building Layers for CNN

#Getting activation function (last Layer) based on the Number of classes
from LastLayer_Softmax import _get_last_layer_units_and_activation
units,activation=_get_last_layer_units_and_activation(15)

#Multi Layer Perceptrons(MLPs)
from MultiLayerPerceptron import mlp_model
MLP=mlp_model(2,15,0.2,np.shape(Train[0]),15)

#Train Model
def train_ngram_model(train_texts, train_labels,val_texts, val_labels,
                      learning_rate=1e-3,
                      epochs=1000,
                      batch_size=128,
                      layers=2,
                      units=64,
                      dropout_rate=0.2):
    """Trains n-gram model on the given dataset.
    
    # Arguments
        data: tuples of training and test texts and labels.
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of Dense layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.

    # Raises
        ValueError: If validation data has label values which were not seen
            in the training data.
    """
    # Get the data.
    #(train_texts, train_labels), (val_texts, val_labels) = data
    print(1)
    # Verify that validation labels are in the same range as training labels.
    from explore_data import get_num_classes
    num_classes = get_num_classes(train_labels)
    unexpected_labels = [v for v in val_labels if v not in range(num_classes)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(
                             unexpected_labels=unexpected_labels))
    print(2)
    # Vectorize texts.
    x_train, x_val = ngram_vectorize(
        train_texts, train_labels, val_texts)
    print(3)
    # Create model instance.
    model = mlp_model(layers=layers,
                                  units=units,
                                  dropout_rate=dropout_rate,
                                  input_shape=x_train.shape[1:],
                                  num_classes=num_classes)
    print(4)
    # Compile model with learning parameters.
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2)]

    # Train and validate model.
    history = model.fit(
            x_train,
            train_labels,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(x_val, val_labels),
            verbose=2,  # Logs once per epoch.
            batch_size=batch_size)

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    model.save('IMDb_mlp_model.h5')
    print('Model Saved')
    return history['val_acc'][-1], history['val_loss'][-1]
#---------------
a,b=train_ngram_model(Train[0],Train[1],Test[0],Test[1])
#--------------------------------------------------------------------------

#Loading Model and Evaluating Performance
import tensorflow as tf
classifierLoad = tf.keras.models.load_model('IMDb_mlp_model.h5')
classifierLoad.evaluate

test=['I want to eat Icecream','Do I need an umbrella','olaf feels its sunny']
train,test=ngram_vectorize(Train[0],Train[1],test)
answer=classifierLoad.predict(test[2])
