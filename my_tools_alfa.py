import pickle
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.stats as measures
import numpy as np

class plot_box_via_neurons:
    def __init__(self, NET:tf.keras.Model, scaler, scaler_out, full_ds: tuple):
        try:
            assert isinstance(NET(1), tf.keras.Model), "No keras model"      #added 1 to NET in oder to check return by fun type
        except:
            assert isinstance(NET, tf.keras.Model), "No keras model"
        self.net_ = NET
        self.input_scaler = scaler
        self.scaler_out = scaler_out
        self.full_ds = full_ds
        self.ttt = []
        self.zzz = []

    def _struc(self):
        len_ = range(len(self.model.get_config()['layers'])-1)
        len_ = list(len_)[1:]
        out = [self.model.layers[x].units for x in len_]
        return out
    
    def compute(self,*args, **kwargs):
        pass

    def plot_box(self):
        pass


#temp solution
class checker_dist_box(plot_box_via_neurons):
    def __init__(self, scaler, scaler_out, full_ds):
        self.input_scaler = scaler
        self.scaler_out = scaler_out
        self.full_ds = full_ds
        self.ttt = []
        self.zzz = [] 
        
    def compute(self,*args, **kwargs):
        self.input_ = kwargs.pop("input_")
        self.output = kwargs.pop("output")
        lista = []
        plt.figure(figsize=(8, 3.5))
        for _ in range(10):
            #tep solution
            
            self.model = tf.keras.Model(inputs = [self.input_], outputs = [self.output])
            self.model.compile(loss = 'MSE',optimizer = 'sgd', metrics = ["MSE"])

            self.model.fit(*args,**kwargs, verbose=0)
            y_pred_ = self.model(self.input_scaler.transform(self.full_ds[0])) 
            y_pred_ = np.ravel(self.scaler_out.inverse_transform(y_pred_))     
            y_true_ = np.ravel(self.full_ds[1])                                                     
            wynik = measures.pearsonr(y_pred_, y_true_)[0] #.round(2)
            lista.append(wynik)
            tf.keras.backend.clear_session()

        print(
            lista, 
            'średnia: ', np.mean(lista).round(3), 
            'max: ', np.max(lista).round(3), 
            'struktura: ', self._struc(), 
            ' !Coef Pearson'
            )
        self.ttt.append(lista)
        self.zzz.append(str(self._struc()))

    @property
    def plot_box(self):
        plt.boxplot(self.ttt, labels = self.zzz)