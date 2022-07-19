#_1 KEEP OUTPUTS IN PANDAS AND PICKLE
import pickle
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.stats as measures
import numpy as np


class hold_data:
    """
    add_out:
    print_:
    save_to_file:
    """
    def __init__(self):
        self.list_: list = []

    def add_out(self, NET,MAX_MSE,TUNER_KERAS,OTHER, EXTRAS1, EXTRAS2) -> list:
        """
        'NET','MAX_MSE','TUNER_KERAS','OTHER', 'EXTRAS#1','EXTRAS#2'
        """
        self.list_.append([NET,MAX_MSE,TUNER_KERAS,OTHER, EXTRAS1, EXTRAS2])
        print('Added: ',
            [NET,MAX_MSE,TUNER_KERAS,OTHER, EXTRAS1, EXTRAS2]
            )
    
    def _make_pd(self) -> None:
        self.pd_ = pd.DataFrame(self.list_, columns = ['NET','MAX_MSE','TUNER_KERAS','OTHER', 'EXTRAS#1','EXTRAS#2'])
        return self.pd_

    def print_(self) -> None:
        print(self._make_pd())
        
    def save_to_file(self, name: str):
        self._make_pd().to_excel(f'{name}.xlsx')
        with open(f'{name}.pickle', 'wb') as f:
            pickle.dump(self.list_, f)

class plot_train_via_neurons:
    def __init__(self, NET:tf.keras.Model, scaler, full_ds: tuple):
        assert isinstance(NET(1), tf.keras.Model), "No keras model"      #added 1 to NET in oder to check return by fun type
        self.net_ = NET
        self.input_scaler = scaler
        self.full_ds = full_ds

    def _struc(self):
        len_ = range(len(self.model.get_config()['layers'])-1)
        out = [self.model.layers[x].units for x in len_]
        return out
    
    def compute(self,*args, **kwargs):
        for NEURONS in [6,7,8,9]:
            lista = []
            plt.figure(figsize=(8, 3.5))
            for _ in range(3):
                self.model = self.net_(NEURONS)
                history = self.model.fit(*args,**kwargs)
                # METRICS
                y_pred_ = np.ravel(self.model(self.input_scaler.transform(self.full_ds[0])))     
                y_true_ = np.ravel(self.full_ds[1])                                                     
                wynik = measures.pearsonr(y_pred_, y_true_)[0] #.round(2)
                lista.append(wynik)
                plt.plot(history.history['loss'])
                #plt.plot(history.history['val_loss'], '--')
                tf.keras.backend.clear_session()
            print(lista, 'średnia: ', np.mean(lista).round(3), 'max: ', np.min(lista).round(3), 'struktura: ', self._struc())
            now = datetime.now()
            current_time = now.strftime("%H_%M_%S")
            plt.text(0,0.05, s = "średnia: {}  max: {}  struktura: {}".format(np.mean(lista).round(3),np.max(lista).round(3),self._struc()) )
            plt.savefig('{}'.format(current_time))


class plot_box_via_neurons:
    def __init__(self, NET:tf.keras.Model, scaler, full_ds: tuple):
        assert isinstance(NET(1), tf.keras.Model), "No keras model"      #added 1 to NET in oder to check return by fun type
        self.net_ = NET
        self.input_scaler = scaler
        self.full_ds = full_ds
        self.ttt = []
        self.zzz = []

    def _struc(self):
        len_ = range(len(self.model.get_config()['layers'])-1)
        out = [self.model.layers[x].units for x in len_]
        return out
    
    def compute(self,*args, **kwargs):
        for NEURONS in [6,7,8,9]:
            lista = []
            plt.figure(figsize=(8, 3.5))
            for _ in range(3):
                self.model = self.net_(NEURONS)
                history = self.model.fit(*args,**kwargs)
                # METRICS
                y_pred_ = np.ravel(self.model(self.input_scaler.transform(self.full_ds[0])))     
                y_true_ = np.ravel(self.full_ds[1])                                                     
                wynik = measures.pearsonr(y_pred_, y_true_)[0] #.round(2)
                lista.append(wynik)
                plt.plot(history.history['loss'])
                #plt.plot(history.history['val_loss'], '--')
                tf.keras.backend.clear_session()
            print(lista, 'średnia: ', np.mean(lista).round(3), 'max: ', np.max(lista).round(3), 'struktura: ', self._struc())
            now = datetime.now()
            current_time = now.strftime("%H_%M_%S")
            plt.text(0,0.05, s = "średnia: {}  max: {}  struktura: {}".format(np.mean(lista).round(3),np.max(lista).round(3),self._struc()) )
            plt.savefig('{}'.format(current_time))
            self.ttt.append(lista)
            self.zzz.append(str(self._struc()))

    def plot_box(self):
        plt.boxplot(self.ttt, labels = self.zzz)


#warning, manual set n neurons,
class checker_dist_box(plot_box_via_neurons):
    def __init__(self,NET, scaler, full_ds):
        super().__init__(NET, scaler, full_ds)

    def compute(self,*args, **kwargs):
        neurons = kwargs.pop("neurons")  
        lista = []
        plt.figure(figsize=(8, 3.5))
        for _ in range(10):
            self.model = self.net_(neurons)         # set n neurons
            self.model.fit(*args,**kwargs)
            y_pred_ = np.ravel(self.model.predict(self.input_scaler.transform(self.full_ds[0])))     
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