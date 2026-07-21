import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
import scipy.signal as signal
from scipy import fft
from IPython.display import display, Math

def freq_analysis(A = 2, noise = 1, start_time = 1, end_time= 10):
    data = pd.DataFrame({"time":[],"difference":[],"difference":[]})
    time_range = np.linspace(start_time, end_time, 1000) * 365
    
    for val in time_range:
        Curve = SYNTHETIC(noise=noise)
        Curve.simple_sin(A = A, b = val)
        Curve.create_gabs()
        #Curve.plot()
        Curve.FourierLombScargle(False)
        #Curve.plot()
        try:
            time = Curve.fourierLombScargle["time"][0]
            strength = Curve.fourierLombScargle["properties"][0]
            differenz = val - time
            data = pd.concat([data, pd.DataFrame({"time": [val],"difference": [differenz], "Strength": [strength]})], ignore_index=True)
        except:
            ...
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))

    axs[0].scatter(data["time"], data["difference"])
    axs[0].set_xlabel("Period time in days")
    axs[0].set_ylabel("Difference to exact value in years")

    axs[1].scatter(data["time"], data["Strength"])
    axs[1].set_xlabel("Period time in days")
    axs[1].set_ylabel("Certainty of detection")

    axs[2].scatter(data["difference"], data["Strength"])
    axs[2].set_xlabel("Difference to exact value in days")
    axs[2].set_ylabel("Certainty of detection")

    plt.tight_layout()
    plt.show()

class SYNTHETIC:
    
    def __init__(self, noise = None, num_points = 2000, timespan = 10):
        self.x = []
        self.y = []
        self.noise = noise
        self.num_points = num_points
        self.timespan = timespan
        self.create_base()
        self.formula = f"" # in days
        self.sin_times = [] #in days
        self.fourierLombScargle = []
        self.fourierFFT = []
    
    def simple_sin(self, A=None, b=None): # A * sin(b * x)
        umrechnung = 24 * 60 * 60
        two_years = 2 * 365 * umrechnung
        twenty_years = 10 * 365 * umrechnung
        b = b*umrechnung
        if A is None: 
            A = np.random.uniform(0.1, 2)
        if b is None:  
            b = np.random.uniform(two_years, twenty_years)

        self.y += A * np.sin((2*np.pi)/b * self.x)
        if len(self.formula) > 1:
            self.formula += "+"
        self.formula += rf"{A:.2f} \cdot \sin\!\left(\frac{{2\pi}}{{{(b/umrechnung):.2f}\,\text{{d}}}} \, x\right)"
        self.sin_times.append(b/umrechnung)
    def create_base(self):
        self.timespan = self.timespan * 365 * 24 * 3600
        self.x = np.linspace(0, self.timespan, self.num_points)
        self.y = np.linspace(0,0,self.num_points)
        if self.noise != None:
            self.y += np.random.normal(0, self.noise, self.num_points)

    def normalize(self):
        y_min = min(self.y)
        self.y -= y_min 
        y_max = max(self.y)
        self.y = self.y / y_max 

    def create_gabs(self):
        map_points = np.sin((2*np.pi)/(2*365 * 24 * 60 * 60) * self.x)
        map_mask = np.abs(map_points) < 0.8
        self.y = self.y[map_mask]
        self.x = self.x[map_mask]
        
    def rolling_mid(self, window_size=30):
        df = pd.DataFrame({"x": self.x, "y": self.y})
        df["y"] = df["y"].rolling(window=window_size*(24*3600), center=True).mean()
        df = df.dropna().reset_index(drop=True)
        self.x = df["x"].values
        self.y = df["y"].values
        
    def plot(self):
        x_years = self.x / (365 * 24 * 60 * 60)

        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure(figsize=(10, 6))

        plt.scatter(
            x_years,
            self.y,
            s=45,                # marker size
            alpha=0.9,           # transparency
            color="#4C78A8",     # marker color
            edgecolor="black",
            linewidth=0.4
        )

        plt.title("Lightcurve", fontsize=15, weight="bold")
        plt.xlabel("Time (years)", fontsize=12)
        plt.ylabel("Flux", fontsize=12)

        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()
        
    
    
    def FourierLombScargle_old(self):
        min_t = 200*60*60*24 # min detectation window in days
        max_t = 2*max(self.x) # max detectation window in days
        frequency, power = LombScargle(self.x, self.y).autopower(minimum_frequency = 1/max_t, maximum_frequency = 1/min_t, samples_per_peak=10)
        fourier = pd.DataFrame({"frequency":frequency,"power":power})
        fourier = fourier[(fourier["frequency"] < 1/min_t) & (fourier["frequency"] > 1/max_t)]
        #plt.plot(fourier["frequency"],fourier["power"])
        #plt.show()
        peaks, properties = signal.find_peaks(fourier["power"], height=0.01)
        peaks = pd.DataFrame({"time":1/frequency[peaks],"frequency":frequency[peaks],"properties":properties["peak_heights"]})
        peaks = peaks.sort_values("properties",ascending=False)
        peaks = peaks.reset_index(drop=True)
        peaks["time"] = peaks["time"]/(365*24*60*60)
        self.fourierLombScargle = peaks
    def FourierLombScargle(self, plot = False):
        self.normalize()
        t = self.x
        y = self.y
        umrechnung = 60*60*24 # von Sekunden in Tage
        t = t /umrechnung
        def rolling_mid(t, y, window=5):
            df = pd.DataFrame({'t': t, 'y': y})
            df = df.sort_values('t')
            df['y_rolling_mid'] = df['y'].rolling(window=window, center=True).mean()
            df = df.dropna()
            return df["t"], df["y_rolling_mid"] 
        t_min = min(t)
        t = t - t_min  
        min_t = 50 #*umrechnung # min detectation window in days
        max_t = 4*max(t) # max detectation window in days
        #t,y = rolling_mid(t, y, window=50)
        # Lomb-Scargle Periodogram
        frequency, power = LombScargle(t, y).autopower(minimum_frequency = 1/max_t, maximum_frequency = 1/min_t, samples_per_peak=10)
        frequency = frequency
        fourier = pd.DataFrame({"frequency":frequency,"power":power})
        fourier = fourier[(fourier["frequency"] < 1/min_t) & (fourier["frequency"] > 1/max_t)]
        peaks, properties = signal.find_peaks(fourier["power"], height=0.05)
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
            print(f"Formula: {display(Math(self.formula))}")
            print(f"Sin times in days: \n{self.sin_times}")
            print(f"Top 3 Peaks:\n{peaks.iloc[0:3]}")
        peaks["time"] = peaks["time"]
        self.fourierLombScargle = peaks
        
    def FourierFFT(self):
        # Annahme: self.x ist in Sekunden, gleichmäßig verteilt
        y = self.y
        N = len(y)
        dt = np.median(np.diff(self.x))  # Abtastintervall in Sekunden

        yf = fft.rfft(y)
        xf = fft.rfftfreq(N, dt)

        power = np.abs(yf)
        fourier = pd.DataFrame({"frequency": xf, "power": power})

        # Peaks finden
        peaks, properties = signal.find_peaks(fourier["power"], height=0.01)
        peak_freqs = xf[peaks]
        peak_times = 1 / peak_freqs
        peaks_df = pd.DataFrame({
            "time": peak_times,
            "frequency": peak_freqs,
            "properties": properties["peak_heights"]
        })
        peaks_df = peaks_df.sort_values("properties", ascending=False).reset_index(drop=True)
        peaks_df["time"] = peaks_df["time"] / (365 * 24 * 60 * 60)  # in Jahre

        self.fourierFFT = peaks_df    
    
    
    
# freq_analysis()    
    
    
    
    
# exit()    # Todo finish the rolling mean 
        
# Curve = SYNTHETIC(noise = 1)
# Curve.simple_sin()
# Curve.plot()
# #Curve.rolling_mid()
# #Curve.plot()
# #Curve.simple_sin()

# Curve.normalize()
# #Curve.create_gabs()
# print(Curve.formula)

# ALE = ANALYSE(Curve.x, Curve.y, Curve.formula)   
# ALE.FourierLombScargle()
# Curve.plot()
        