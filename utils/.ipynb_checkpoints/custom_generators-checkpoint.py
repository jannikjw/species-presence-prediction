import tensorflow as tf
import threading
import numpy as np
from GLC.data_loading.common import load_patch

class RGBImageGenerator(tf.keras.utils.Sequence) :
  
    def __init__(self, obs_ids, labels, batch_size, data_path) :
        self.obs_ids = obs_ids
        self.labels = labels
        self.batch_size = batch_size
        self.data_path = data_path

        # to make the generator thread safe 
        self.lock = threading.Lock()

    def __len__(self) :
        return (np.floor(len(self.obs_ids) / float(self.batch_size))).astype(int)
  
    # returns one batch
    def __getitem__(self, idx) :
        X_batch = list()
        y_batch = list()

        for i in range(idx * self.batch_size, (idx+1) * self.batch_size):
            if i >= len(self.obs_ids): break
            
            patch = load_patch(self.obs_ids[i], self.data_path, data='rgb')
            X_batch.append(patch[0])
            y_batch.append(self.labels[i])

        with self.lock:
            return np.asarray(X_batch), np.array(y_batch)

class FullSatelliteGenerator(tf.keras.utils.Sequence) :
  
    def __init__(self, obs_ids, labels, batch_size, data_path) :
        self.obs_ids = obs_ids
        self.labels = labels
        self.batch_size = batch_size
        self.data_path = data_path
        
        # to make the generator thread safe 
        self.lock = threading.Lock()

    def __len__(self) :
        return (np.floor(len(self.obs_ids) / float(self.batch_size))).astype(int)
  
    # returns one batch
    def __getitem__(self, idx) :
        X_batch = list()
        y_batch = list()

        for i in range(idx * int(self.batch_size/2), (idx+1) * int(self.batch_size/2)):
            if i >= len(self.obs_ids): break
            
            rgb, near_ir, landcover, altitude = load_patch(self.obs_ids[i], self.data_path)

            ni = near_ir.reshape(256, 256, 1)
            lc = landcover.reshape(256, 256, 1)
            alt = altitude.reshape(256, 256, 1)

            patch = np.concatenate((ni, lc, alt), axis=2)
            
            # add rgb as one observation
            X_batch.append(rgb)
            y_batch.append(self.labels[i])
            
            # rest of data as second observation
            X_batch.append(patch)
            y_batch.append(self.labels[i])

        with self.lock:
            return np.asarray(X_batch), np.array(y_batch)

class Patches_Generator_CNN(tf.keras.utils.Sequence) :
  
    def __init__(self, obs_ids, labels, batch_size, data_path) :
        self.obs_ids = obs_ids
        self.labels = labels
        self.batch_size = batch_size
        self.data_path = data_path
        
        # to make the generator thread safe 
        self.lock = threading.Lock()

    def __len__(self) :
        return (np.floor(len(self.obs_ids) / float(self.batch_size))).astype(int)
  
    # returns one batch
    def __getitem__(self, idx) :
        X_batch = list()
        X_env_batch = list()
        y_batch = list()

        for i in range(idx * self.batch_size, (idx+1) * self.batch_size):
            if i >= len(self.obs_ids): break
            
            rgb, near_ir, landcover, altitude = load_patch(self.obs_ids[i], self.data_path)

            ni = near_ir.reshape(256, 256, 1)
            lc = landcover.reshape(256, 256, 1)
            alt = altitude.reshape(256, 256, 1)

            patch = np.concatenate((rgb, ni, lc, alt), axis=2)
            
            X_batch.append(patch)
            y_batch.append(self.labels[i])
            
            X_env_batch.append(tabular_train[self.obs_ids[i]].values)
            
        with self.lock:
            return {'input_1': np.asarray(X_batch), 'input_2': np.asarray(X_env_batch)}, np.asarray(np.array(y_batch))


# +
class MultimodalTransformerGenerator(tf.keras.utils.Sequence) :
  
    def __init__(self, obs_ids, labels, batch_size, data_path, env_df) :
        self.obs_ids = obs_ids
        self.labels = labels
        self.batch_size = batch_size
        self.data_path = data_path
        self.env_df = env_df
        
        # to make the generator thread safe 
        self.lock = threading.Lock()

    def __len__(self) :
        return (np.floor(len(self.obs_ids) / float(self.batch_size))).astype(int)
  
    # returns one batch
    def __getitem__(self, idx) :
        X_batch = list()
        X_env_batch = list()
        y_batch = list()

        for i in range(idx * self.batch_size, (idx+1) * self.batch_size):
            if i >= len(self.obs_ids): break
            
            rgb, near_ir, landcover, altitude = load_patch(self.obs_ids[i], self.data_path)

#             ni = near_ir.reshape(256, 256, 1)
#             lc = landcover.reshape(256, 256, 1)
#             alt = altitude.reshape(256, 256, 1)

#             patch = np.concatenate((rgb, ni, lc, alt), axis=2)
            
            X_batch.append(rgb)
            y_batch.append(self.labels[i])
            
            X_env_batch.append(self.env_df[i])
            
        with self.lock:
            return {'input_1': np.asarray(X_batch), 'input_2': np.asarray(X_env_batch)}, np.asarray(np.array(y_batch))
