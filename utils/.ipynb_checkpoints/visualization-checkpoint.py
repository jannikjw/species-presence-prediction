# +
import matplotlib.pyplot as plt
import numpy as np
from utils.augmentation import data_augmentation_transformer, data_augmentation_cnn
from GLC.data_loading.common import load_patch
from utils.data_loader import DataLoader

def visualize_augmentation(ids_list, data_path, num_images=4, contrast=0.1, flip='horizontal', rotation=0.02):
    plt.figure(figsize=(16, 16))
    images = []
    for i in range(num_images):
        images.append(np.array(load_patch(ids_list[i], data_path, data='rgb')).reshape(256, 256, 3))

    for i in range(num_images):
        image = np.array(load_patch(ids_list[i], data_path, data='rgb')).reshape(256, 256, 3)
        images.append(data_augmentation_cnn(256, contrast=contrast, flip=flip, rotation=rotation)(image).numpy())

    for i, image in enumerate(images):
        ax = plt.subplot(num_images, num_images, i + 1)
        plt.imshow(image.astype("uint8"))
        plt.axis("off")
        
def visualize_sat(ids_list, image, data_path):
    plt.figure(figsize=(16, 16))
    images = []
    rgb, near_ir, landcover, altitude = load_patch(ids_list[image], data_path)
    images.append(np.array(rgb).reshape(256, 256, 3))
    images.append(near_ir)
    images.append(landcover)
    images.append(altitude)
    
    for i, image in enumerate(images):
        ax = plt.subplot(1, 4, i + 1)
        plt.imshow(image.astype("uint8"))
        plt.axis("off")
        
def plot_training_results(results, top_k=5):
    fig, axs = plt.subplots(1, 2, figsize=(10,6))
    axs[0].plot(range(len(results.history['loss'])), results.history['loss'], label="Training Loss")
    axs[0].plot(range(len(results.history['val_loss'])), results.history['val_loss'], label="Validation Loss")
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[1].plot(range(len(results.history['val_accuracy'])), results.history['val_accuracy'], label='Validation accuracy')
    axs[1].plot(range(len(results.history['val_top-{}-accuracy'.format(top_k)])), results.history['val_top-{}-accuracy'.format(top_k)], label='Top-{} Validation accuracy'.format(top_k))
    axs[1].legend()
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    plt.show()
