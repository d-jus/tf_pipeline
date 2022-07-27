#_1 KEEP OUTPUTS IN PANDAS AND PICKLE
import pickle
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.stats as measures
import numpy as np

class Hold_data:
    """
    add_out:
    print_:
    save_to_file:
    """
    def __init__(self):
        self.list_: list = []

    def add_out(self, model, history) -> list:
        """
        model: tf.Model
        history: hisotory of fiting model

        Returns: list
        """
        self.list_.append([
            round(history.history['loss'][-1],3),
            model.layers[1].kernel_initializer.__class__.__name__,
            model.layers[1].activation.__name__,
            model.optimizer.__class__.__name__
        ])
        print('Added:', self.list_[-1])
    
    def _make_pd(self) -> None:
        self.pd_ = pd.DataFrame(self.list_, columns = ['loss','ker_ini','act_fn','opt'])
        return self.pd_

    def print_(self) -> None:
        print(self._make_pd())
        
    def save_to_file(self, name: str):
        self._make_pd().to_excel(f'{name}.xlsx')
        with open(f'{name}.pickle', 'wb') as f:
            pickle.dump(self.list_, f)


class Plot_train_via_neurons:
    def __init__(self, NET:tf.keras.Model, scaler, scaler_out, full_ds: tuple):
        try:
            assert isinstance(NET(1), tf.keras.Model), "No keras model"      #added 1 to NET in oder to check return by fun type
        except:
            assert isinstance(NET, tf.keras.Model), "No keras model"
        self.net_ = NET
        self.input_scaler = scaler
        self.scaler_out = scaler_out
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
                y_pred_ = self.model(self.input_scaler.transform(self.full_ds[0]))
                y_pred_ = np.ravel(self.scaler_out.inverse_transform(y_pred_))     
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
                y_pred_ = self.model(self.input_scaler.transform(self.full_ds[0]))
                y_pred_ = np.ravel(self.scaler_out.inverse_transform(y_pred_))
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
class Checker_dist_box(Plot_box_via_neurons):
    def __init__(self,NET, scaler, scaler_out, full_ds):
        super().__init__(NET, scaler, scaler_out, full_ds)

    def compute(self,*args, **kwargs):
        print(kwargs)
        if "neurons" in kwargs.keys():
            _neurons = kwargs.pop("neurons")
        else:
            _neurons = False             
         
        lista = []
        plt.figure(figsize=(8, 3.5))
        for _ in range(10):
            
            if _neurons:
                self.model = self.net_(_neurons)     # set n neurons 
            else:
                self.model = self.net_

            print(self.model.layers[1].get_weights())

            self.model.fit(*args,**kwargs)
            y_pred_ = self.model(self.input_scaler.transform(self.full_ds[0])) 
            y_pred_ = np.ravel(self.scaler_out.inverse_transform(y_pred_))     
            y_true_ = np.ravel(self.full_ds[1])                                                     
            wynik = measures.pearsonr(y_pred_, y_true_)[0] #.round(2)
            lista.append(wynik)
            tf.keras.backend.clear_session()
            tf.keras.backend.reset_uids()

            print(self.model.layers[1].get_weights())

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