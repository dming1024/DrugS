
# -*- coding: utf-8 -*-
"""
Created on 2024.10
@author: demin
Description: prediction of drug response based on gene expression profiling
"""

# Import necessary modules
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import argparse  # For handling command-line arguments
import logging   # For logging


from scipy.stats import zscore
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from tensorflow.keras import layers, callbacks,Model
from tensorflow.keras.layers import Dense, Dropout

class DrugS:
    # Constructor method
    def __init__(self):
        self.name = 'DrugS'  # instance variable
        self.version = 'v0.0'    # instance variable

    # Method to display information
    def display_info(self):
        print(f"Name: {self.name}, version: {self.version}")


    def transform_data(self,gene_matrix_files_in,features_file):
        sc_count=pd.read_csv(gene_matrix_files_in,index_col=0).T
        features=np.load(features_file, allow_pickle=True)
        result_df = sc_count.reindex(columns=features, fill_value=0)
        
        #zscore
        result_df=result_df.apply(zscore,axis=0)
        #na FILL
        result_df = result_df.fillna(0)
        return result_df        
    
    def reduction_dimension(self,result_df,autoencoder_model):
        encoding_dim= 30
        tf.random.set_seed(42)
        np.random.seed(42)
        input_dim=19193
        input_layer = keras.Input(shape=(input_dim,))
        encoded = Dense(2048, activation='elu')(input_layer)
        encoded = Dense(encoding_dim, activation='relu')(encoded)
        decoded = Dense(2048, activation='elu')(encoded)
        decoded = Dense(input_dim, activation='relu')(decoded)
        autoencoder = Model(input_layer, decoded)
        encoder = Model(input_layer, encoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        early_stopping = callbacks.EarlyStopping(
            min_delta=0.001, # minimium amount of change to count as an improvement
            patience=10, # how many epochs to wait before stopping
            restore_best_weights=True,
        )
        autoencoder=keras.models.load_model(autoencoder_model) 
        result_df_sc = encoder.predict(result_df)
        result_df_sc = pd.DataFrame(result_df_sc, columns=[f'encoded_{i+1}' for i in range(30)])
        return result_df_sc

    def load_pre_model(self,all_molecule_morgan,dnn_model):
        fp=pd.read_csv(all_molecule_morgan,index_col=0,dtype={0: str})
        model=keras.models.load_model(dnn_model)
        return(fp,model)


def get_compound_predicted_ic50(fp,result_df_sc,model):
    row_to_add_df = pd.DataFrame([fp] * len(result_df_sc)).reset_index(drop=True)
    result = pd.concat([row_to_add_df, result_df_sc], axis=1)
    tmp=model.predict(result)
    res=pd.Series([m[0] for m in tmp])
    return res



def main():
    parser = argparse.ArgumentParser(description="Description of your script.")
    parser.add_argument('--expression', type=str, help="gene expression (gene symbol[row] x samples[column]), TPM, FPKM suggested")
    args = parser.parse_args()

    drugs=DrugS()
    drugs.display_info()
    result_df=drugs.transform_data(args.expression,'./dataset/features.npy')
    result_df_sc=drugs.reduction_dimension(result_df,'./models/autoencoder_model.h5')
    fp,model=drugs.load_pre_model('./dataset/all_molecule_morgan_compound_name.csv','./models/dnn_model_basedon_encoder30_gdsc1_gdsc2/')
    
    new_df=fp.apply(get_compound_predicted_ic50,axis=1,args=(result_df_sc,model))
    new_df.columns=result_df.index
    new_df.to_csv("prediction_results.csv")


if __name__ == "__main__":
    main()





