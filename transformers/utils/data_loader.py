# +
from pathlib import Path
import pandas as pd
import numpy as np
import random

random.seed(2)

class DataLoader():
    '''
    Class to handle all data-related tasks
    '''
    def __init__(self, data_path="./geolifeclef-2022-lifeclef-2022-fgvc9/"):
        # let's load the data from file
        self.data_path = Path(data_path)
        
        df_obs_fr = pd.read_csv(self.data_path / "observations" / "observations_fr_train.csv", sep=";", index_col="observation_id")
        df_obs_us = pd.read_csv(self.data_path / "observations" / "observations_us_train.csv", sep=";", index_col="observation_id")
        self.df_obs = pd.concat((df_obs_fr, df_obs_us))

    def subset_labels(self, set_type, num_labels=10):
        '''
        Select only a subset of labels
        Returns: observation ids and labels
        '''
        if set_type not in ['train', 'val']: 
            return 'You need to choose one of the set types "train" or "val".'

        df_obs = self.df_obs
        # select the ids from the given subset
        obs_ids = df_obs.index[df_obs["subset"] == set_type].values
        labels = df_obs.loc[obs_ids]["species_id"].values       

        obs_list = list()

        # iterate over a subset of the labels
        counter = 0
        for label in np.unique(labels)[:num_labels]:
            # for each label, retrieve all corresponding observation ids
            obs = df_obs[(df_obs["species_id"] == label) & (df_obs["subset"] == set_type)].index.values
            obs_list.append(obs)

        # we now have a numpy array of all observation ids corresponding to this subset of labels
        observation_ids = np.concatenate(obs_list)

        # obtain the labels in the right order 
        labels = df_obs.loc[observation_ids]["species_id"].values

        return observation_ids, labels
    
    def subset_labels_dense(self, num_labels=10, min_obs=1000, max_obs=2000):
        '''
        Select only a subset of labels that have a certain number of observations.
        '''
        subset_size = 0
        obs_train = list()
        obs_test = list()
        obs_val = list()

        df_obs = self.df_obs
        # select the ids from the given subset
        labels = df_obs["species_id"].values       

        # iterate over a subset of the labels
        for label in set(labels):
            if (subset_size >= num_labels): break
                
            obs = df_obs.index[(df_obs["species_id"] == label)]

            if(len(obs) >= min_obs and len(obs) <= max_obs):
                # for each label, retrieve all corresponding observation ids
                train_obs = set(df_obs[(df_obs["species_id"] == label) & (df_obs["subset"] == 'train')].index.values)
                val_obs = df_obs[(df_obs["species_id"] == label) & (df_obs["subset"] == 'val')].index.values
                
                # take 10% of train set as test set
                test = random.sample(train_obs, int(len(obs)/10))
                train = set(obs) - set(test)
                obs_test.append(list(test))
                obs_train.append(list(train))
                subset_size += 1
                
                obs_val.append(val_obs)
                
        # we now have a numpy array of all observation ids corresponding to this subset of labels
        train_ids = np.concatenate(obs_train)
        val_ids = np.concatenate(obs_val)
        test_ids = np.concatenate(obs_test)

        # obtain the labels in the right order         
        y_train = df_obs.loc[train_ids]["species_id"].values
        y_val = df_obs.loc[val_ids]["species_id"].values        
        y_test = df_obs.loc[test_ids]["species_id"].values
        
        # remap labels to new subset size
        map_labels = {}
        for i, label in enumerate(set(y_train)):
            map_labels[label] = i
        
        for i, label in enumerate(y_train):
            y_train[i] = map_labels[label]
        for i, label in enumerate(y_val):
            y_val[i] = map_labels[label]
        for i, label in enumerate(y_test):
            y_test[i] = map_labels[label]

        return train_ids, y_train, val_ids, y_val, test_ids, y_test
    
    def load_environmental_data(self, train_ids, val_ids, test_ids):
        df_obs = self.df_obs
        
        df_obs_bio = pd.read_csv(self.data_path / "pre-extracted" / "environmental_vectors.csv", sep=";", index_col="observation_id")
        df_environmental_vars = pd.read_csv(self.data_path / "metadata" / "environmental_variables.csv", sep=";", index_col="name")

        # collect gps data for selected observations
        gps_data_train = df_obs.loc[train_ids][['latitude', 'longitude']]
        gps_data_val = df_obs.loc[val_ids][['latitude', 'longitude']]
        gps_data_test = df_obs.loc[test_ids][['latitude', 'longitude']]

        # collect biological and pedologic data
        bio_data_train = df_obs_bio.loc[train_ids]
        bio_data_val = df_obs_bio.loc[val_ids]
        bio_data_test = df_obs_bio.loc[test_ids]
        
        # join environmental data and GPS coordinates
        df_train = bio_data_train.join(gps_data_train)
        df_val = bio_data_val.join(gps_data_val)
        df_test = bio_data_test.join(gps_data_test)

        return df_train, df_val, df_test
    
    def get_data_path(self):
        return self.data_path
    
    def set_data_path(self, data_path):
        self.data_path = Path(path)
    
    def rename_environmental_table(self, df):
        # rename columns of joined table with meaningful descriptions 
        df_environmental_vars = pd.read_csv(self.data_path / "metadata" / "environmental_variables.csv", sep=";", index_col="name")
        environmental_vars = list(df_environmental_vars['description'])
        environmental_vars.append('latitude')
        environmental_vars.append('longitude')

        df.columns = environmental_vars

        return df

def transform_table_to_sentences(df):
    # convert numerical values to strings 
    df = df.astype(str)

    # add column name as token before every cell value
    for col in df.columns:
        df[col] = df[col].map(lambda x: "<{}> {}".format(col, x))

    # create "sentences" from table
    df = list(df.stack().groupby(level=0).apply(' '.join))

    return df

def tokenize(sentences, tokenizer, max_length):
    input_ids, input_masks, input_segments = [], [], []
    for sentence in sentences:
        inputs = tokenizer.encode_plus(sentence, add_special_tokens = True, max_length = max_length, pad_to_max_length = True, truncation=True, return_attention_mask = True, return_token_type_ids = True)
        input_ids.append(inputs['input_ids'])
        input_masks.append(inputs['attention_mask'])
        input_segments.append(inputs['token_type_ids'])
    return np.asarray(input_ids, dtype = "int32"), np.asarray(input_masks, dtype = "int32"), np.asarray(input_segments, dtype = "int32")
    
if __name__ == "__main__":
    loader = DataLoader()
# -


