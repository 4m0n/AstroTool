import pandas as pd 
import config 
from pathlib import Path
from ..base import LightCurve
from astropy.coordinates import SkyCoord
import astropy.units as u

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
    df = read_data_from_jd(path / name)
    
    curve.data = df
    curve.original_name = name
    curve.explanations = "loaded from bachelor thesis data"
    curve.original_path = str(path / name)
    return curve    
            
def run():
    path = config.RAW_DATA_DIR / "light_curves"
    
    path_df_coords = path / "name_id.csv"
    df_coords = pd.read_csv(path_df_coords)
    
    data = []
    for file in path.iterdir():
        if file.is_file() and file.suffix == ".csv":
            curve = LightCurve()
            df = read_data_from_jd(path / file)
            if df is None:
                continue
            
            
            curve.data = df
            curve.original_name = file.stem
            curve.explanations = "loaded from bachelor thesis data"
            curve.original_path = str(path / file)
            file_id = file.stem.split('-')[0]
            mask = df_coords['ID'].astype(str) == file_id
            if file_id in df_coords["ID"].astype(str).values:
                ra, dec = df_coords[mask][['ra', 'dec']].values[0]
                curve.coordinates = SkyCoord(f"{ra} {dec}", unit=(u.hourangle, u.deg), frame="icrs")
            data.append(curve)  
            
                
    return data
            