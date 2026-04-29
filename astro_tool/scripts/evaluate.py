import pandas as pd
import numpy as np
import config
from astropy.timeseries import LombScargle
import scipy.signal as signal
import matplotlib.pyplot as plt
import scripts.base as base
import time
from tqdm import tqdm
import inspect
import scripts.process as process

value = config.VALUE_CALCULATION

def delta(y,err):
    sum = 0
    length = len(y)    
    for i in range (length):
        sum += err[i]**2
    try:
        sum = np.sqrt(sum/length)
    except:
        sum = 0
    return sum # nicht quadriert

class parameter_calculations:
    @staticmethod
    def calculate_parameters(only_new = True, data = None):
        if data == None:
            collection = base.load_processed_data(None, False)
            # === Get all functions in this class except this one ===
            func_name = inspect.currentframe().f_code.co_name
            funcs = []
            for name in dir(parameter_calculations):
                if name == func_name or name.startswith("_"):
                    continue
                attr = getattr(parameter_calculations, name)
                funcs.append(attr)

            # === Call all functions on the data===
            for data in tqdm(collection,"Calculating parameters"):
                if len(data.data) < 50:
                    continue
                for param in funcs:
                    data = param(data = data, only_new = False)

                data.save()
        else:
            func_name = inspect.currentframe().f_code.co_name
            funcs = []
            for name in dir(parameter_calculations):
                if name == func_name or name.startswith("_"):
                    continue
                attr = getattr(parameter_calculations, name)
                funcs.append(attr)

            # === Call all functions on the data===
            for param in funcs:
                data = param(data = data, only_new = False)
            data.save()  
        
        
    @staticmethod
    def FourierLombScargle(data, only_new = False,plot = False):
        if only_new == True:
            if not data.frequency.empty:
                return data
        data.normalize(normalize=True)     
        def rolling_mid(t, y, window=5):
            df = pd.DataFrame({'t': t, 'y': y})
            df = df.sort_values('t')
            df['y_rolling_mid'] = df['y'].rolling(window=window, center=True).mean()
            df = df.dropna()
            return df["t"], df["y_rolling_mid"] 
        
        df = data.data
        t,y = df["JD"].values, df[value].values
        t_min = min(t)
        t = t - t_min  
        try:
            t,y = rolling_mid(t, y, window=15)
            if len(y) < 10:
                return data
        except:
            print(f"KAPUTT: {y} - len: {len(df)} \n{df}")
        y = y - 0.5  
        min_t = 10 # min detectation window in days
        max_t = 2*max(t) # max detectation window in days

        # Lomb-Scargle Periodogram
        frequency, power = LombScargle(t, y).autopower(minimum_frequency = 1/max_t, maximum_frequency = 1/min_t, samples_per_peak=10)
        fourier = pd.DataFrame({"frequency":frequency,"power":power})
        fourier = fourier[(fourier["frequency"] < 1/min_t) & (fourier["frequency"] > 1/max_t)]
        peaks, properties = signal.find_peaks(fourier["power"], height=0.2)
        peaks = pd.DataFrame({"time":1/frequency[peaks],"frequency":frequency[peaks],"properties":properties["peak_heights"]})
        peaks = peaks.sort_values("properties",ascending=False)
        peaks = peaks.reset_index(drop=True)
        
        if plot:
            fig, (ax_t, ax_w, ax_f) = plt.subplots(3, 1, constrained_layout=True)
            ax_t.plot(t, y, 'b+')
            ax_w.scatter(1.0/fourier["frequency"],fourier["power"])
            ax_f.plot(fourier["frequency"],fourier["power"])
            ax_w.set_xlim(min_t,max_t)
            ax_f.set_xlim(1/min_t,1/max_t)
            ax_f.set_xlabel('Period duration [1/days]')
            ax_t.set_ylabel('Normalized amplitude')
            ax_t.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax_w.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.get_current_fig_manager().full_screen_toggle()
            plt.show()
        data.parameters.frequency = peaks.iloc[0:3]
        return data
    @staticmethod
    def Fractional_variation(data,only_new = False):
        if only_new == True:
            if data.frequency.Fvar != None:
                return data
        data.normalize(normalize=True)    
        y = data.data[value].values.copy()
        err = data.data[value + " Error"].values.copy()
        if np.std(y)**2 < delta(y,err)**2:
            data.parameters.Fvar = np.nan
            return data
        activity = np.sqrt((np.std(y)**2-delta(y,err)**2)) / np.mean(y)
        data.parameters.Fvar = activity
        return data
    @staticmethod
    def peak_to_peak_amplitudes(data,only_new = False): # not normalized otherwise always the same 
        if only_new == True:
            if data.frequency.R != None:
                return data
        data.normalize(normalize=False)
        curve = data.data
        if min(curve[value]) <= 0:
            return data
        # if curve[value].min() <= 0:
        #     print(f"Warning: Galaxy {data.get_name()} has negative values")
        #     data.parameters.R = 0
        #     return data
        R = curve[value].max() / curve[value].min()
        if abs(curve[value].min()) < 1 and abs(curve[value].max()) > 20 or abs(curve[value].min()) < 0.1:
            data.parameters.R = 0
            return data
        data.parameters.R = R
        data.normalize(normalize=True)
        return data
    @staticmethod
    def slope(data,only_new = False):
        if only_new == True:
            if data.frequency.slope != None:
                return data
        data.normalize(normalize=True) 
        return data
        
    @staticmethod
    def standard_values(data,only_new = False):
        if only_new == True:
            if data.frequency.mean != None:
                return data
        if data.currently_loaded != "processed":
            print(f"Warning: standard_values should be calculated on processed data, but currently_loaded is {data.currently_loaded}")
        data.normalize(normalize=True) 
        curve = data.data
        data.parameters.std = curve[value].std()
        data.parameters.mean = curve[value].mean()
        data.parameters.median = curve[value].median()
        return data
        

def evaluate_specific():
    EVAL = parameter_calculations()
    data = [base.LightCurve.load("34360082638")]
    for i in tqdm(range(len(data))):
        data[i] = process.start(data[i])
        if len(data[i].data) < 10:
            print(f"Trash: {data[i].get_name()}")
            continue
        #data[i] = EVAL.FourierLombScargle(data[i])
        data[i] = EVAL.Fractional_variation(data[i])
        data[i].last_evaluated = time.time()
        data[i].save()
        return data[i]

def evaluate_all_preprocessed():
    EVAL = parameter_calculations()
    data = base.load_processed_data_list()    
    for i in tqdm(range(len(data))):
        if len(data[i].data) < 10:
            print(f"Trash: {data[i].get_name()}")
            continue
        if "661434889104" in str(data[i].original_name):
            print(f"KURVE max: {data[i].data['Flux'].max()} min: {data[i].data['Flux'].min()}")
        data[i].normalize(normalize=True)
        data[i] = EVAL.FourierLombScargle(data[i])
        data[i] = EVAL.Fractional_variation(data[i])
        data[i] = EVAL.peak_to_peak_amplitudes(data[i])
        data[i] = EVAL.slope(data[i])
        data[i] = EVAL.standard_values(data[i])
        data[i].last_evaluated = time.time()
        data[i].normalize(normalize=False)
        data[i].save()
        
        
#? R unterliegt bedinungen: min und max einschränkung für Werte die nicht in die Tausend gehen