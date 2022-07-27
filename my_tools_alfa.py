import pickle
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.stats as measures
import numpy as np

class Plot_box_via_neurons:
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
        len_ = range(len(self.model.get_config()['layers']))
        len_ = list(len_)[1:]
        out = [self.model.layers[x].units for x in len_]
        return out
    
    def compute(self,*args, **kwargs):
        pass

    def plot_box(self):
        pass


#temp solution
class Checker_dist_box(Plot_box_via_neurons):
    def __init__(self, scaler, scaler_out, full_ds):
        self.input_scaler = scaler
        self.scaler_out = scaler_out
        self.full_ds = full_ds
        self.ttt = []
        self.zzz = [] 
        
    def compute(self,*args, **kwargs):
        self.neurons_ = kwargs.pop("neurons_") # useless
        self.function_ = kwargs.pop("function_") # useless
        lista = []
        plt.figure(figsize=(8, 3.5))
        for _ in range(10):
            #tep solution
            self.model = None
            tf.keras.backend.clear_session()

            input_ = tf.keras.layers.Input(shape=(12,))
            hidden1_ = tf.keras.layers.Dense(self.neurons_, activation = self.function_)(input_)
            output = tf.keras.layers.Dense(1)(hidden1_)
            self.model = tf.keras.Model(inputs = [input_], outputs = [output])
            self.model.compile(loss = 'MSE',optimizer = 'sgd', metrics = ["MSE"])

            self.model.fit(*args,**kwargs, verbose=0)
            y_pred_ = self.model(self.input_scaler.transform(self.full_ds[0])) 
            y_pred_ = np.ravel(self.scaler_out.inverse_transform(y_pred_))     
            y_true_ = np.ravel(self.full_ds[1])                                                     
            wynik = measures.pearsonr(y_pred_, y_true_)[0] #.round(2)
            lista.append(wynik)

            

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

class Num1(Checker_dist_box):
    """
    Inicjalizacja: LeCuna
    Funkcja aktywacji: SELU
    Normalizacja: -
    Regularyzacja: 
    Optymalizator: RMSProp; Nadam
    Harmonogram uczenia: -
    """
    def __init__(self, scaler, scaler_out, full_ds):
        super().__init__(scaler, scaler_out, full_ds)
        
    def compute(self,*args, **kwargs):
        self.neurons_ = kwargs.pop("neurons_") # useless
        self.function_ = kwargs.pop("function_") # useless
        lista = []
        plt.figure(figsize=(8, 3.5))
        for _ in range(10):
            #tep solution
            self.model = None
            tf.keras.backend.clear_session()

            input_ = tf.keras.layers.Input(shape=(12,))
            hidden1_ = tf.keras.layers.Dense(self.neurons_, activation = self.function_, kernel_initializer = tf.keras.initializers.LecunNormal())(input_)
            output = tf.keras.layers.Dense(1, kernel_initializer = tf.keras.initializers.LecunNormal())(hidden1_)
            self.model = tf.keras.Model(inputs = [input_], outputs = [output])
            self.model.compile(loss = 'MSE',optimizer = 'Nadam', metrics = ["MSE"])

            self.model.fit(*args,**kwargs, verbose=0)
            y_pred_ = self.model(self.input_scaler.transform(self.full_ds[0])) 
            y_pred_ = np.ravel(self.scaler_out.inverse_transform(y_pred_))     
            y_true_ = np.ravel(self.full_ds[1])                                                     
            wynik = measures.pearsonr(y_pred_, y_true_)[0] #.round(2)
            lista.append(wynik)

            

        print(
            lista, 
            'średnia: ', np.mean(lista).round(3), 
            'max: ', np.max(lista).round(3), 
            'struktura: ', self._struc(), 
            ' !Coef Pearson'
            )
        self.ttt.append(lista)
        self.zzz.append(str(self._struc()))