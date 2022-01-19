import numpy as np
import pandas as pd
from src.constants import *
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.model_selection  import train_test_split
from sklearn.utils import shuffle

class Net:
    """
    Net class is the class of the tensorflow neural network in order to find number of retweet. 
    It uses a preprocess table from preprocessing.py.
    """

    def __init__(self, table, N_NEURONS):
        self.table = table

        self.N_NEURONS = N_NEURONS ### number of neurons in the hidden layer
        
    
    def table_split(self):
        self.X = np.array(self.table.loc[:, self.table.columns != retweet_count])
        self.y = np.array(self.table[retweet_count], ndmin = 2).T
        self.dim_data = self.X.shape[1]
        print(self.table.columns)

    def data_scaling(self,bert=False):
        if not bert : 
            self.table.drop(columns = [f"{i}" for i in range(50)])

        for column in self.table.columns:
            if column not in non_scalable_col:                
                self.table[column] = self.table[column]/(self.table[column].max())
                # self.table[column] = np.exp(self.table[column]/(self.table[column].max()))
                # self.table[column] = self.table[column]/(self.table[column].sum()) 

    def build_model(self):
        """
        We build the model.
        """
        # First, we build the object "model"
        model = tf.keras.Sequential(name="Neuronal_Network")
        # Then we add the first layer 
        model.add(layers.Dense(units = self.N_NEURONS, input_shape=(self.dim_data,), bias_initializer="glorot_normal", kernel_initializer="glorot_normal"
        # ,kernel_regularizer=regularizers.l1(1e-3)
        ))
        model.add(layers.Activation(tf.math.sigmoid))
        #We had the hidden layer
        model.add(layers.Dense(units = self.N_NEURONS, bias_initializer="glorot_normal", kernel_initializer="glorot_normal"
        # ,kernel_regularizer=regularizers.l1(1e-3)
        ))
        model.add(layers.ReLU())
        model.add(layers.Dense(units = self.N_NEURONS, bias_initializer="glorot_normal", kernel_initializer="glorot_normal"
        # ,kernel_regularizer=regularizers.l1(1e-3)
        ))
        model.add(layers.ReLU()) 

        #We add the output layer
        model.add(layers.Dense(units = 1, bias_initializer="glorot_normal", kernel_initializer="glorot_normal"))
        self.model = model 
        self.model.summary() 

    def loss_MAE(self,y_real,y_pred):
        """
        Using the absolute mean error.
        """
        return tf.reduce_mean(tf.abs(y_real - y_pred))

    def compute_train(self,X_batch,y_batch):
        """
        Train step function. 
        """

        with tf.GradientTape() as tape:
            y_pred = self.model(X_batch)
            loss_value = self.loss_MAE(y_pred, y_batch)
    
        gradients = tape.gradient(loss_value, self.model.trainable_variables)
        return loss_value, gradients

    def run_model(self, EPOCHS = 100 ,BATCH_SIZE = 10000, plot = True, N_EVAL = 5,saving_path=None, eval = True, load_weights = None, bert = False ):
        np.random.seed(123)
        self.data_scaling(bert)
        self.table_split()
        self.build_model()
        if load_weights != None :
            self.model.load_weights(load_weights)
        learning_rate = 0.01
        self.train_list_loss = []
        self.test_list_loss =[]
        seed_counter = 0
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        data_set = np.array(self.table).astype(np.float32)
        dataset_train,dataset_test = train_test_split(data_set,train_size = 0.7)
        
        min_decrease_rate = 0.0005       

        cost_hist = []
        print("Table split")  
        # Boucle d'entraînement
        for epoch in range(1, EPOCHS+1):
            
            seed_counter +=1
            data_train = shuffle(dataset_train, random_state=seed_counter)
            data_batches = tf.data.Dataset.from_tensor_slices(data_train).batch(BATCH_SIZE)
            for i,data in enumerate(data_batches):
                np_data = data.numpy()
                X_batch,y_batch = np_data[:,1:].astype(np.float64),np.array(np_data[:,0],ndmin=2).T.astype(np.float64)
                loss_value, gradients = self.compute_train(X_batch,y_batch)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                
            if (epoch) % N_EVAL == 0 or epoch == 1: 
                print(epoch)
                loss_value, _ = self.compute_train(self.X, self.y)  # sur toutes les données
                self.train_list_loss.append(loss_value)    
                y_pred = self.model(dataset_test[:,1:])#.numpy().astype(int)  
                res = np.array(dataset_test[:,0],ndmin=2).T-y_pred  
                res = np.mean(np.abs(res))
                self.test_list_loss.append(res)
                print(f"The test loss is {res} and the train loss is {loss_value}")
                if res < 126 : 
                    break 
                

        if eval : 
            # Compute predictions
            df = pd.read_csv('data/preprocessed_evaluation.csv')
            if not bert: 
                df.drop(columns = [f"{i}" for i in range(50)],inplace = True)
            for col in df.columns : 
                if col not in non_scalable_col:
                    df[col]/=(df[col].max())
            X = np.array(df)
            y_pred = self.model(X).numpy()#.astype(int) 
            eval_data = pd.read_csv("data/evaluation.csv")
            eval_data["NoRetweets"] = y_pred.astype(int)
            eval_data = eval_data[[ID,"NoRetweets"]]
            eval_data.rename(columns = {ID:"TweetID"}, inplace = True)
            eval_data.to_csv("data/predictions.csv",index = False)


        plt.plot(self.train_list_loss,label = 'train_loss')
        plt.plot(self.test_list_loss, label = 'test_loss')
        plt.title("Bert Neural Network")
        plt.savefig('neural.png')
        plt.legend()
        plt.show()
        if saving_path != None :
            self.model.save_weights(saving_path)
        print("Entraînement terminé !")
