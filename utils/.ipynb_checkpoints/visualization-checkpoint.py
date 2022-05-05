# +
import matplotlib.pyplot as plt
import numpy as np
from utils.augmentation import data_augmentation_transformer, data_augmentation_cnn
from GLC.data_loading.common import load_patch
from utils.data_loader import DataLoader

def visualize_augmentation(ids_list, data_path, num_images=4):
    plt.figure(figsize=(16, 16))
    images = []
    for i in range(num_images):
        images.append(np.array(load_patch(ids_list[i], data_path, data='rgb')).reshape(256, 256, 3))

    for i in range(num_images):
        image = np.array(load_patch(ids_list[i], data_path, data='rgb')).reshape(256, 256, 3)
        images.append(data_augmentation_cnn(256)(image).numpy())

    for i, image in enumerate(images):
        ax = plt.subplot(num_images, num_images, i + 1)
        plt.imshow(image.astype("uint8"))
        plt.axis("off")
        
def plot_training_results(results):
    fig, axs = plt.subplots(1, 2, figsize=(10,6))
    axs[0].plot(range(len(results.history['loss'])), results.history['loss'])
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[1].plot(range(len(results.history['val_accuracy'])), results.history['val_accuracy'], label='Validation accuracy')
    axs[1].plot(range(len(results.history['val_top-10-accuracy'])), results.history['val_top-10-accuracy'], label='Top-10 Validation accuracy')
    axs[1].legend()
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    plt.show()
