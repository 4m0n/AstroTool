import pandas as pd 
import config
from pathlib import Path
from ..base import LightCurve


def read_data_from_jd(filepath):
    
    with open(filepath, 'r') as file:
        for i, line in enumerate(file):
            if line.startswith("JD"):
                start_line = i
                break
        else:
            return None
    return pd.read_csv(filepath, skiprows=start_line)



def load_orignal_data(name):
    path = config.RAW_DATA_DIR / "light_curves"
    curve = LightCurve()
    if ".csv" not in name:
        name += ".csv"
    df = read_data_from_jd(path / name)
    
    curve.data = df
    curve.original_name = name
    curve.explanations = "loaded extra from single folder"
    curve.original_path = str(path / name)
    return curve    
            


def start(name, new_name = None):
    data = load_orignal_data(name)
    data.preprocess(new_name = None)

    data.evalute()
    data.save()
    return data