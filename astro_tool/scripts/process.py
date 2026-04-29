import pandas as pd
from os import listdir
import os
from os.path import isfile, join
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import numpy as np
import math
import time
from tqdm import tqdm
from scipy import optimize
import yaml
import config
import time
value = config.VALUE_CALCULATION
mid = config.ROLLING_WINDOW

def pre_cleaning(data):
    if "JD"  in data.columns and "Date" not in data.columns:
        data["Date"] = pd.to_datetime(data['JD'], origin='julian', unit='D')
    data.dropna(subset=[value, "Date"], inplace=True)
    data = data[(data[f"{value} Error"] < 10)]
    return data

def add_filter_to_cams(file):
    for idx, row in file.iterrows():
        file.loc[idx, "Filter_Camera"] = f'{row["Filter"]}-{row["Camera"]}'
    return file

def neumann_cam_shift(data,curve):
    
    #find overlapp 
    start_overlap = max(data["JD"].min(), curve["JD"].min())
    end_overlap = min(data["JD"].max(), curve["JD"].max())
    if data["JD"].max() < curve["JD"].min(): # ! Verschiebt Kurve mit mean falls es keine Überlagerung gibt
        len_data = len(data["JD"])
        len_curve = len(curve["JD"])
        points = 40 
        if min(len_data,len_curve) < 40:
            points = min(len_data,len_curve)
        mean_existing = data[value][-points:].mean()
        mean_shift_curve = curve[value][:points].mean()
        shift_mean = mean_existing - mean_shift_curve
        curve[value] = curve[value] + shift_mean
        return curve,shift_mean
    if len(curve["JD"]) < 2:
        len_data = len(data["JD"])
        points = 40 
        if len_data < 40:
            points = len_data
        mean_existing = data[value][-points:].mean()
        mean_shift_curve = curve[value].mean()
        shift_mean = mean_existing - mean_shift_curve
        curve[value] = curve[value] + shift_mean
        return curve,shift_mean
            
    if end_overlap < start_overlap:
        return curve,0
    
    main_curve = data[(data['JD'] >= start_overlap) & (data['JD'] <= end_overlap)].copy()
    fit_curve = curve[(curve['JD'] >= start_overlap) & (curve['JD'] <= end_overlap)].copy()
    main_curve.reset_index(drop=True, inplace=True)
    fit_curve.reset_index(drop=True, inplace=True)
    
    if len(main_curve["JD"]) < 1: # falls die Kurve genau in eine Lücke ohne Daten fällt
        len_overlap = len(data[(data['JD'] <= start_overlap)])
        if len_overlap < 10:
            len_overlap = len(data[(data['JD'] <= start_overlap)])
        else :
            len_overlap = 10
        mean_not_overlapping_curve = data.loc[(data['JD'] <= start_overlap), value][-len_overlap:].mean()
        mean_shift_curve = curve[value].mean()
        shift_mean = mean_not_overlapping_curve - mean_shift_curve
        curve[value] = curve[value] + shift_mean
        return curve,shift_mean
    #NGC4395 als beispiel für eine Lücke
    
    fit_curve_backup = fit_curve.copy()
    mean_plot_curve = fit_curve.copy() #! löschen
    #R = np.arange(-20,20,0.01) # ! Funktion für die Range schreiben um minimum schneller zu finden
    T = []
    go = True
    shift = 0
    R = []
    step = 1
    T_save = 10e10
    change = True # True == <= und False == >
    while True:
        T.append(0)
        R.append(0)
        fit_curve = fit_curve_backup.copy()
        fit_curve[value] = fit_curve[value] + shift
        calculate_curve = pd.concat([fit_curve,main_curve])
        calculate_curve.sort_values(by='JD', inplace=True)
        calculate_curve.reset_index(drop=True, inplace=True)
        
        for i in range(len(calculate_curve)-1):
            T[-1] += (calculate_curve[value][i] - calculate_curve[value][i+1])**2 
        if abs(step) < 0.01 or (T[-1] == 0 and abs(shift) >= 100):
            R[-1] = shift
            break
        
        if T[-1] > T_save:
            step = -step/2
        T_save = T[-1]
        R[-1] = shift
        shift = shift + step
            
            
    shift = R[T.index(min(T))]
    # TODO ZUM TESTEN MEAN BERECHNET -> SOLLTE NOCH GELÖSCHT WERDEN
    mean1 = main_curve[value].mean()
    mean2 = mean_plot_curve[value].mean()
    mean_diff = mean1 - mean2
    mean_plot_curve[value] = mean_plot_curve[value] + mean_diff
    # ============================================================
    if False:
        for shift in R:
            T.append(0)
            fit_curve = fit_curve_backup.copy()
            fit_curve[value] = fit_curve[value] + shift
            calculate_curve = pd.concat([fit_curve,main_curve])
            calculate_curve.sort_values(by='JD', inplace=True)
            calculate_curve.reset_index(drop=True, inplace=True)
            
            for i in range(len(calculate_curve)-1):
                T[-1] += (calculate_curve[value][i] - calculate_curve[value][i+1])**2
            
        shift = R[T.index(min(T))]
    
    all_data = pd.concat([data,curve])
    all_data.sort_values(by='JD', inplace=True)
    fit_curve[value] = fit_curve_backup[value] + shift
    curve[value] = curve[value] + shift
    return curve, shift





def remove_outliers_mad(data, threshold=3):
    file = data.data
    file = add_filter_to_cams(file)
    backup_curve = file.copy()
    # EVLT ÄNDERN??
    #if (file[f"{value} Error"] > 10).sum() < 100:
        
    # delete rows without camera info (could also stay)
    file['Camera'].replace('', np.nan)
    file = file.dropna(subset=['Camera'])
    file.reset_index(drop=True, inplace=True)


    position = len(file)
    cams = file["Filter_Camera"].unique().copy()
    for _ in range(1):
        for c in cams:
            df = file[file.Filter_Camera == c].reset_index(drop=True)
            df = df.sort_values(by='Date').reset_index(drop=True)

        
            if True:    
                mean = df[value].rolling(window=mid, center=True).mean() # median besser als mean?
                #print(mean.to_string())
                mean_mean = mean.mean()
                std = df[value].rolling(window=mid, center=True).std()
                std_mean = std.mean()
                mean = np.where(np.isnan(mean), mean_mean, mean)
                std = np.where(np.isnan(std), std_mean, std)

                for i in reversed(range(0,len(df))):
                    #if df[value][i] > mean[i] + mean[i]*0.02 or df[value][i] < mean[i] - mean[i]*0.02:
                    if df.at[i,value] > ((mean[i] + std[i])*1.01) or df.at[i,value] < ((mean[i] - std[i])*0.99):
                        df.drop(i, inplace=True)
            df.reset_index(drop=True, inplace=True)  # Indizes nach dem Löschen zurücksetzen
            file = pd.concat([file, df], ignore_index=True)
            
        file.drop(index=range(0, position), inplace=True)
        file.reset_index(drop=True, inplace=True)
    # ==== g Filter verschieben ==== 

    #move("Filter",["V","g"])
    #move("Camera",cameras)

    # === calculating deleted points ===
    org_dates = [val for val in backup_curve["JD"]]    
    new_dates = [val for val in file["JD"]]    
    deleted_dates = []
    for val in org_dates:
        if val not in new_dates:
            temp_value = backup_curve[backup_curve["JD"] == val][value].values
            deleted_dates.append([val,temp_value])
    data.cuts.outliers = deleted_dates
    data.data = file
    return data


def shift_cam_and_filters(data,analyse = False,cuts = True):
    file = data.data
    cameras = file["Filter_Camera"].unique() 
    if len(cameras) <= 1: #! eine kurve reicht eigentlich auch schon
        return file
    min = []
    for c in cameras:
        min.append(file.loc[file["Filter_Camera"] == c, "Date"].min())

    #Kameras sortieren
    for i in range(len(min)):   
        for j in range(len(min)-i-1):
            if min[j] > min[j+1]:
                min[j], min[j+1] = min[j+1], min[j]
                cameras[j], cameras[j+1] = cameras[j+1], cameras[j]
    
    main_curve = file[file["Filter_Camera"] == cameras[0]].copy()
    analyse_cuts = pd.DataFrame()
    analyse_cuts_conditions = pd.DataFrame()
    for i in range(1,len(cameras)):
        #print(f"cam: {cameras[i]} nr: {i}")
        fit_curve = file[file["Filter_Camera"] == cameras[i]].copy()
        # Check ob es größere Lücken oder Sprünge gibt -> Dann Lichtkurve weiter unterteilen # ! Wenn ein Cluster stark verschoben ist -> wird als seperate Kurve behandelt
        if cuts:
            fit_curve.sort_values(by='Date', inplace=True)
            fit_curve.reset_index(drop=True, inplace=True)
            #curve_splitter = pd.DataFrame(columns=["cut", "mean", "std","timediff"], dtype='float')
            curve_splitter = pd.DataFrame({
                "cut": pd.Series(dtype='int64'),
                "mean": pd.Series(dtype='float'),
                "std": pd.Series(dtype='float'),
                "timediff": pd.Series(dtype='timedelta64[ns]')
            })
            curve_splitter = curve_splitter.astype({'cut': 'int64'})         
            start = 0
            # ==== Bedingung 1 
            for k in range(len(fit_curve["Date"])-1):
                if (fit_curve.iloc[k+1]["Date"] - fit_curve.iloc[k]["Date"] > pd.Timedelta(days=30)) or (abs(fit_curve.iloc[k+1][value] - fit_curve.iloc[k][value]) > fit_curve[value].std()): # Maximale Lückengröße
                    if len(fit_curve[value][start:k]) <2: #! entfernen? (wenn in einem einzelnen Zeitraum weniger als 2 Werte vorhanden sind)
                        start = k+1
                        continue
                    curve_splitter = pd.concat([curve_splitter, pd.DataFrame([{"cut":k,"mean":fit_curve[value][start:k].mean(),"std":fit_curve[value][start:k].std(),"timediff":fit_curve.iloc[k+1]["Date"] - fit_curve.iloc[k]["Date"]}])], ignore_index=True)
                    data.cuts.cases.append(fit_curve.iloc[k]["Date"]) # start date not end date
                    start = k+1
            # ==== Bedingung 2
            # avg_days = 4
            # for i in range(avg_days,len(fit_curve["Date"])-avg_days - 1):  
            #     mean = fit_curve[value][i-avg_days:i].mean()
            #     if abs(mean - fit_curve[value][i+1:i+1+avg_days].mean()) > (fit_curve[value][i-avg_days:i].std() - fit_curve[value][i+1:i+1+avg_days].std()).mean():
            #         for k in range(i,len(fit_curve["Date"])):


            # ================
                                        
            mean_std = curve_splitter["std"].mean()
            new_curves = pd.DataFrame(columns = ["cut_start"])
            
            start_data = {"cut_start":[0,len(fit_curve[value])]}
            new_curves = pd.DataFrame(start_data)
            def steigung(file):
                def linear(x,m,b):
                    return m*x+b
                curve = file.copy()

                numeric_index = []
                for i in curve["Date"]:
                    numeric_index.append(i.timestamp())
                    
                # ===== FIT ======
                if len(numeric_index) < 2:
                    return 0
                min_num = np.min(numeric_index)
                diff = np.max(numeric_index) - min_num
                numeric_index = numeric_index - min_num
                numeric_index = numeric_index/diff
                try:
                    params, params_covariance = optimize.curve_fit(linear, numeric_index, curve[value].values, p0=[1,0.5],maxfev=100000) # m*x+b
                    m,b  = params[0],params[1]
                except:
                    return 0
                return m
            
            temp_df = pd.DataFrame()    
            for k in range(1,len(curve_splitter["cut"])):
                # ==== Bedingung 1
                m =  abs(steigung(fit_curve.loc[curve_splitter['cut'][k-1]:curve_splitter['cut'][k]])) < 0.5
                bedingung = (curve_splitter["std"][k-1] < mean_std*1.2) and (curve_splitter["std"][k] < mean_std*1.1) and ((curve_splitter["mean"][k] > curve_splitter["mean"][k-1]*1.05) or (curve_splitter["mean"][k] < curve_splitter["mean"][k-1]*0.95))

                #bedingung 2 für große abstände + großen sprung
                bedingung2 = (curve_splitter["timediff"][k-1] > pd.Timedelta(days=60) and (abs(curve_splitter["mean"][k] / curve_splitter["mean"][k - 1]) > 2 or abs(curve_splitter["mean"][k] / curve_splitter["mean"][k - 1]) < 0.5))
                
                # Versuch mit noramalisierung:
                m =  abs(steigung(fit_curve.loc[curve_splitter['cut'][k-1]:curve_splitter['cut'][k]])) < 0.5/73.5
                bedingung = (curve_splitter["std"][k-1] < mean_std*2) and (curve_splitter["std"][k] < mean_std*1.5) and ((curve_splitter["mean"][k] > curve_splitter["mean"][k-1]*1.1 * 1.109435) or (curve_splitter["mean"][k] < curve_splitter["mean"][k-1]*0.9 * 1.109435))

                #bedingung 2 für große abstände + großen sprung
                bedingung2 = False # (curve_splitter["timediff"][k-1] > pd.Timedelta(days=60) and (abs(curve_splitter["mean"][k] / (curve_splitter["mean"][k - 1]* 1.109435)) > 2 or abs(curve_splitter["mean"][k] / (curve_splitter["mean"][k - 1]* 1.109435) < 0.5)))
                
                
                
                # ===============================
                #bedingung = (curve_splitter["std"][i-1] < mean_std*2) and (curve_splitter["std"][i] < mean_std*1.5) and ((curve_splitter["mean"][i] > curve_splitter["mean"][i-1]*1.1) or (curve_splitter["mean"][i] < curve_splitter["mean"][i-1]*0.9))
                #bedingung 2 für große abstände + großen sprung
                #bedingung2 = (curve_splitter["timediff"][i-1] > pd.Timedelta(days=60) and (abs(curve_splitter["mean"][i] / curve_splitter["mean"][i - 1]) > 2 or abs(curve_splitter["mean"][i] / curve_splitter["mean"][i - 1]) < 0.5))
                
                
                
                # print(f"EHH Bed1:\n{bedingung}\nBed2:{bedingung2}\n")
                row = {
                    "start_index": curve_splitter['cut'][k-1],
                    "end_index": curve_splitter['cut'][k],
                    "std_prev": curve_splitter["std"][k-1],
                    "std_next": curve_splitter["std"][k],
                    "mean_prev": curve_splitter["mean"][k-1],
                    "mean_next": curve_splitter["mean"][k],
                    "mean_std": mean_std,
                    "timediff": curve_splitter["timediff"][k-1],
                    "fit_curve_slice": steigung(fit_curve.loc[curve_splitter['cut'][k-1]:curve_splitter['cut'][k]]),
                    "m":m,
                    "bedingung": bedingung,
                    "bedingung2": bedingung2,  
                    "final": (m and bedingung) or bedingung2,
                }

                temp_df = pd.concat([temp_df, pd.DataFrame([row])], ignore_index=True)    
                                            
                if (m and bedingung) or bedingung2:
                    new_curves = pd.concat([new_curves, pd.DataFrame([{"cut_start":curve_splitter["cut"][k-1]+1}])], ignore_index=True)
                        
            filename = "analyse_cuts_test1_no_norm.csv"

            if os.path.exists(filename):
                existing_df = pd.read_csv(filename, index_col=0)
                result_df = pd.concat([existing_df, temp_df], ignore_index=True)
            else:
                result_df = temp_df

            #result_df.to_csv(filename)
                
            new_curves.sort_values(by='cut_start', inplace=True)
            new_curves.reset_index(drop=True, inplace=True)
            for i in range(1,len(new_curves["cut_start"])):
                fit_curve2,shift = neumann_cam_shift(main_curve,fit_curve.iloc[new_curves["cut_start"][i-1]:new_curves["cut_start"][i]].copy())
                temp = {"start_date":fit_curve.iloc[new_curves["cut_start"][i-1]]["Date"],
                        "end_date":fit_curve.iloc[new_curves["cut_start"][i]-1]["Date"],
                        "shift":shift,
                        }
                data.cuts.cuts.append(temp) 
                main_curve = pd.concat([main_curve, fit_curve2], ignore_index=True)
                main_curve.sort_values(by='Date', inplace=True)
            global current_cuts
            current_cuts = len(new_curves["cut_start"]) - 2
            
        else:  
            fit_curve,_ = neumann_cam_shift(main_curve,fit_curve)
            main_curve = pd.concat([main_curve, fit_curve], ignore_index=True)
            main_curve.sort_values(by='Date', inplace=True)
    if analyse:
        return analyse_cuts,analyse_cuts_conditions
    return main_curve



def start(data, analyse = False):
    data.data = pre_cleaning(data.data) # erstellt datum und JD, index separate 
    data.normalize(normalize = True) # normalize
    ### ========== PREPROCESSING ALGORITHMS ========== ###
    data = remove_outliers_mad(data) # removes points around each camera
    data.normalize(normalize = False)
    data.normalize(normalize = True)
    data.data = shift_cam_and_filters(data,analyse) #braucht am längsten
    if analyse:
        return data.data
    ### ================================ ###
    data.normalize(normalize = False) # back to orignal scale

    return data.data
