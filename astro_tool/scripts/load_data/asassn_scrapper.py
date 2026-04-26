
from pyasassn.client import SkyPatrolClient
client = SkyPatrolClient()

import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import pandas as pd



class SkyPatrol:
    def __init__(self):
        from pyasassn.client import SkyPatrolClient
        self.client = SkyPatrolClient()
        


    @staticmethod
    def get_galaxies(ra_deg = 270, dec_deg = -88, radius = 0.1, download = True):
        query_str = f"""
            SELECT
                asas_sn_id, ra_deg, dec_deg
                FROM stellar_main
                WHERE DISTANCE(ra_deg, dec_deg, {ra_deg}, {dec_deg}) <= {radius}
            """
        results = client.adql_query(query_str, download=download,threads=10)
        results = SkyPatrol.load_curves(results)
        return results
    
    @staticmethod
    def load_curves(lcs):
        names = list(lcs.ids)
        curves = []
        for name in names:
            temp = lcs[name].data
            if len(temp) < 10:
                continue
            temp.rename(columns={"jd":"JD","flux":"Flux","flux_err":"Flux Error","mag":"Mag","mag_err":"Mag Error","camera":"Camera","phot_filter":"Filter"}, inplace=True)   
            #temp["Date"] = pd.to_datetime(temp['JD'], origin='julian', unit='D')
            curves.append(temp)
        return curves