import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
import scipy.signal as signal
from scipy import fft


def freq_analysis():
    data = pd.DataFrame({"time":[],"difference":[],"difference":[]})
    time_range = np.linspace(1, 10, 100) * 365 * 24 * 60 * 60
    
    for val in time_range:
        Curve = SYNTHETIC(noise=1)
        Curve.simple_sin(b = val)
        Curve.create_gabs()
        ANE = ANALYSE(Curve.x, Curve.y, Curve.formula)
        ANE.FourierLombScargle()
        #Curve.plot()
        try:
            time = ANE.fourierLombScargle["time"][0]
            strength = ANE.fourierLombScargle["properties"][0]
            differenz = val/(365 * 24 * 60 * 60) - time
            data = pd.concat([data, pd.DataFrame({"time": [val/(365 * 24 * 60 * 60)],"difference": [differenz], "Strength": [strength]})], ignore_index=True)
        except:
            ...

    fig, axs = plt.subplots(3, 1, figsize=(12, 8))

    axs[0].scatter(data["time"], data["difference"])
    axs[0].set_xlabel("Period time in years")
    axs[0].set_ylabel("Difference to exact value in years")

    axs[1].scatter(data["time"], data["Strength"])
    axs[1].set_xlabel("Period time in years")
    axs[1].set_ylabel("Certainty of detection")

    axs[2].scatter(data["difference"], data["Strength"])
    axs[2].set_xlabel("Difference to exact value in years")
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
        self.formula = f""
    
    def simple_sin(self, A=None, b=None): # A * sin(b * x)
        umrechnung = 365 * 24 * 60 * 60
        two_years = 2 * umrechnung
        twenty_years = 10 * umrechnung
        if A is None: 
            A = np.random.uniform(0.1, 2)
        if b is None:  
            b = np.random.uniform(two_years, twenty_years)

        self.y += A * np.sin((2*np.pi)/b * self.x)
        self.formula += f"frequency: {(b/umrechnung):.2f} years ---- {A:.2f} * sin((2*pi)/{(b/umrechnung):.2f} * x)\n"
    
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
        plt.scatter(self.x/(365 * 24 * 60 * 60),self.y)
        plt.show()
        
    
class ANALYSE:
    def __init__(self, x, y, formula = ""):
        self.x = x
        self.y = y
        self.formula = formula
        self.fourierLombScargle = []
        self.fourierFFT = []
    
    def rolling_mid(self):
        ...
    
    def FourierLombScargle(self):
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
    
    
    
#freq_analysis()    
    
    
    
    
        
Curve = SYNTHETIC(noise = 1)
Curve.simple_sin()
Curve.plot()
Curve.rolling_mid()
Curve.plot()
exit()    # Todo finish the rolling mean 
#Curve.simple_sin()

Curve.normalize()
#Curve.create_gabs()
print(Curve.formula)

ALE = ANALYSE(Curve.x, Curve.y, Curve.formula)   
ALE.FourierLombScargle()
Curve.plot()
        