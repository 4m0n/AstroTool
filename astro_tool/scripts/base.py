import pandas as pd
import time
import pickle
import config
from pathlib import Path
from .process import start
import numpy as np
import scripts.evaluate as evalute
import traceback
from astropy.coordinates import SkyCoord
import astropy.units as u


# sollte als generator geschrieben werden i guess
def load_processed_data(amount = None, random = True):
    save_class_path = config.INTERIM_DATA_DIR
    directories = [val for val in save_class_path.iterdir()]
    if random:
        np.random.shuffle(directories)
    if (amount != None ) and (amount < len(directories)):
        directories = directories[:amount]
        
    data = []
    for val in directories:
        if ".pickle" in str(val):
            data.append(LightCurve.load(val.name))
    return data





class Cuts:
    def __init__(self):
        self.cases = []
        self.cut = {
                "start_date":None,
                "end_date":None,
                "shift":None
                }
        self.outliers = []
        self.cuts = []
        

class Parameters:
    def __init__(self):
        self.Fvar = None
        self.frequency = pd.DataFrame()
        self.slope = None
        self.R = None
        self.std = None
        self.mean = None
        self.median = None
        
    @staticmethod
    def get_parameters(data = None):
        if data != None:
            for name, value in data.__dict__.items():
                print(f"{name}: {value}")
            print(data)
            return data
        else: 
            collection = load_processed_data(None, False)
            df = pd.DataFrame()
            for val in collection: 
                params = val.parameters
                entry = {}
                for name, value in params.__dict__.items():
                    if name == "std":
                        continue
                    entry[name] = value
                entry["name"] = val.get_name()
                df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
            df.to_csv("parameters_overview.csv")
            return df
                    
                
                
class LightCurve:
    
    def ensure_attributes(self):
        # Erzeuge eine frische Instanz, um alle aktuellen Attribute zu bekommen
        fresh = LightCurve()
        for key, value in fresh.__dict__.items():
            if not hasattr(self, key):
                setattr(self, key, value)
    
    def __init__(self):
        self.data = pd.DataFrame()
        self.added = time.time()
        self.last_processed = None
        self.last_evaluated = None
        self.explanations = ""
        
        self.original_name = ""
        self.name = ""
        self.coordinates = None
        
        self.original_path = ""
        self.class_path = ""
        self.new_path = ""
        
        self.VALUE_CALCULATION = "Flux"
        
        self.orignal_length = 0
        self.new_length = 0
        
        self.normalized = False # true or false
        self.normalize_shift = 0.0
        self.normalize_factor = 0.0
        
        self.currently_loaded = ""
        self.parameters = Parameters()
        self.cuts = Cuts()
        
        

    def show(self):
        string = "====== Light Curve Info ======\n"
        string += f"All calculations performed on {self.VALUE_CALCULATION}\n"
        string += f"original name: {self.original_name} - new name {self.name}\n"
        string += f"added: {self.added}\nExplanation:{self.explanations}\n"
        string += f"last processed: {self.last_proccessed} - last evaluated = {self.last_evaluated}\n"
        string += f"Original length: {self.orignal_length} - New length: {self.new_length} diff: {self.orignal_length - self.new_length}\n"
        string += f"Data preview:\n{self.data.head()}\n"
        string += "==============================\n"
        print(string)
        
    def preprocess(self, new_name = None, anayse = False):
        self.orignal_length = len(self.data)
        if new_name != None:
            self.name = new_name
        if anayse:
            return start(self,anayse)
        else:
            self.data = start(self)
        self.last_proccessed = time.time()
        #self.new_path = config.INTERIM_DATA_DIR / f"{self.get_name()}.pickle"
        self.VALUE_CALCULATION = config.VALUE_CALCULATION
        self.new_length = len(self.data)
        self.currently_loaded = "processed"
     
    def evalute(self):
        evalute.parameter_calculations.calculate_parameters(only_new=False, data = self) 
        
    def get_name(self):
        name = self.name
        try:
            if name == "":
                name = self.original_name
        except:
            name = ""
        if ".csv" in name:
            name = name.replace(".csv", "")
        return str(name)
    
    def plot_before_after(self):
        from plots import compare_before_after_preprocessing
        compare_before_after_preprocessing(self)    
        
        
    def normalize(self, normalize = None):
        value = self.VALUE_CALCULATION
        if self.normalized == False and normalize != False:
            shift = self.data[value].min()
            self.data[value] = self.data[value] - shift
            scale = self.data[value].max()
            self.data[value] = self.data[value] / scale
            self.data[value + " Error"] = self.data[value + " Error"] / scale
            self.normalize_shift = shift
            self.normalize_factor = scale
            self.normalized = True
            
        elif normalize != True:
            self.data[value] = self.data[value] * self.normalize_factor
            self.data[value + " Error"] = self.data[value + " Error"] * self.normalize_factor
            self.data[value] = self.data[value] + self.normalize_shift
            self.normalize_factor = 0.0
            self.normalize_shift  = 0.0
            self.normalized = False
        
        
    def load_all_proccessed():
        save_class_path = config.INTERIM_DATA_DIR  
        data_list = []
        for file in save_class_path.iterdir():
            if file.suffix == ".pickle":
                obj = LightCurve.load(file.name)  
                if obj is not None:
                    data_list.append(obj)
        return data_list
    
    @staticmethod 
    def load(name):
        save_class_path = config.INTERIM_DATA_DIR
        if ".pickle" in name:
            name = name.replace(".pickle", "")
        path = save_class_path / f"{name}.pickle" 
        #self.class_path = str(path)
        if not path.exists():
            print("File does not exist:", path)
            directory = save_class_path.iterdir()
            for val in directory:
                if name in str(val):
                    path = save_class_path / val 
                    print(f"New Path: {path}")
                    break
            else:
                return None
        with open(path, "rb") as file:
            try:
                data = pickle.load(file) 
            except Exception as e:
                print(f"ERROR loading {path}:\n{e}")
                traceback.print_exc()
                return None
        data.ensure_attributes()    
        return data
    def save(self):
        save_class_path = config.INTERIM_DATA_DIR
        name = self.get_name()
        path = save_class_path / f"{name}.pickle"
        self.class_path = str(path)
        with open(path, "wb") as file:
            pickle.dump(self, file)   
            
            
    
    @staticmethod
    def read_data_from_thesis(filepath):
        with open(filepath, 'r') as file:
            for i, line in enumerate(file):
                if line.startswith("JD"):
                    start_line = i
                    break
            else:
                return None
        return pd.read_csv(filepath, skiprows=start_line)
    
    def load_orignal_thesis_data(self, keep = False):
        self.currently_loaded = "orignal"  
        path = self.original_path 
        #path = config.RAW_DATA_DIR / "light_curves" /(self.get_name() + ".csv")     
           
        data = self.read_data_from_thesis(path)  
        if "JD"  in data.columns and "Date" not in data.columns:
            data["Date"] = pd.to_datetime(data['JD'], origin='julian', unit='D')
        if keep:   
            self.data = data
        else:
            return data
        
    def load_processed_data(self):
        if ".pickle" in str(self.new_path):
            self.data = LightCurve.load(self.get_name()).data
            self.currently_loaded = "processed"
        elif self.new_path and Path(self.new_path).exists():
            self.data = pd.read_csv(self.new_path)
            self.currently_loaded = "processed"
        else:
            self.preprocess()
            self.currently_loaded = "processed"
        return self.data
    
    def save_csv(self):
        save_path = config.PROCESSED_DATA_DIR
        name = self.get_name()
        path = save_path / f"{name}.csv"
        data = self.data[["JD", "Date",self.VALUE_CALCULATION, f"{self.VALUE_CALCULATION} Error", "Filter","Camera","Quality"]]
        data.to_csv(path, index=False)
    @staticmethod
    def load_asas_sn(name = None, id = None, rec = None, deg = None, radius = None):
        # id has to be implemented
        from scripts.load_data.asassn_scrapper import SkyPatrol
        if rec == None:
            rec, deg, radius = 270, -88, 0.1
        curves = SkyPatrol.get_galaxies(ra_deg = rec, dec_deg = deg, radius = radius, download = True)
        liste = []
        for val in curves:
            curve = LightCurve()
            curve.data = val
            curve.original_name = val.asas_sn_id
            curve.name = val.asas_sn_id
            curve.explanations = f"loaded from ASAS-SN SkyPatrol around RA: {rec} DEC: {deg} with radius {radius} deg"
            curve.original_path = f"ASAS-SN SkyPatrol id: {id}"
            liste.append(curve)
        return liste
    

        
        
        
        