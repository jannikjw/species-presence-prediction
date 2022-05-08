# +
from tensorflow import keras
from tensorflow.keras import layers
from utils.custom_layers import SwapAxes

def data_augmentation_transformer(output_size, contrast=0.1, flip='horizontal', rotation=0.02):
    '''
    Augment data for Transformers

    Input Shape = (X, X, 3)
    Output Shape = (image_size, 3, image_size)
    '''
    data_augmentation = keras.Sequential(
        [
            layers.Resizing(output_size, output_size),
            layers.Rescaling(1./255),
            SwapAxes(-1, 1),
            layers.RandomContrast(contrast),
            layers.RandomFlip(flip),
            layers.RandomRotation(factor=rotation),
        ],
        name="data_augmentation",
    )
    
    return data_augmentation

def data_augmentation_cnn(output_size, contrast=0.2, flip='horizontal', rotation=0.02):
    '''
    Augment data for CNNs
    
    Input Shape = (X, X, channels)
    Output Shape = (image_size, image_size, channels)
    '''
    data_augmentation = keras.Sequential(
        [
            layers.Resizing(output_size, output_size),
            layers.RandomContrast(0.2),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
        ],
        name="data_augmentation",
    )
    return data_augmentation
