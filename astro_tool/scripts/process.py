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
                full_std = df[value].std()
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




def shift_cam_and_filters(data, analyse=False, cuts=True):
    file = data.data

    # === Kameras sortieren ===
    cam_min_dates = file.groupby("Filter_Camera")["Date"].min()
    cameras = cam_min_dates.sort_values().index.tolist()

    if len(cameras) <= 1:
        return file

    main_curve = file[file["Filter_Camera"] == cameras[0]].copy()

    analyse_rows = []
    analyse_conditions = []

    for cam in cameras[1:]:
        fit_curve = file[file["Filter_Camera"] == cam].copy()

        if not cuts:
            fit_curve, _ = neumann_cam_shift(main_curve, fit_curve)
            main_curve = pd.concat([main_curve, fit_curve], ignore_index=True)
            main_curve.sort_values(by='Date', inplace=True)
            continue

        # === Vorbereitung (NumPy Arrays) ===
        fit_curve.sort_values(by="Date", inplace=True)
        fit_curve.reset_index(drop=True, inplace=True)

        dates = fit_curve["Date"].values
        values = fit_curve[value].values

        std_all = np.std(values)

        # === Step 1: Cuts finden ===
        cuts_idx = []
        start = 0

        for k in range(len(dates) - 1):
            time_gap = dates[k+1] - dates[k]
            value_jump = abs(values[k+1] - values[k])

            if (time_gap > np.timedelta64(30, 'D')) or (value_jump > std_all):
                if (k - start) >= 2:
                    cuts_idx.append((start, k))
                start = k + 1

        if (len(values) - start) >= 2:
            cuts_idx.append((start, len(values)-1))

        # === Segment Statistiken ===
        segments = []
        for (s, e) in cuts_idx:
            seg_values = values[s:e+1]
            seg_dates = dates[s:e+1]

            # normierte Zeit
            t = (seg_dates.astype('datetime64[s]').astype(float))
            t = (t - t.min())
            if t.max() > 0:
                t = t / t.max()

            # schnelle lineare Regression
            if len(t) > 1:
                m = np.polyfit(t, seg_values, 1)[0]
            else:
                m = 0

            segments.append({
                "start": s,
                "end": e,
                "mean": np.mean(seg_values),
                "std": np.std(seg_values),
                "slope": m,
                "timediff": seg_dates[-1] - seg_dates[0]
            })

        if len(segments) == 0:
            continue

        mean_std = np.mean([seg["std"] for seg in segments])

        # === Step 2: Segmente zusammenführen ===
        new_starts = [segments[0]["start"]]

        for i in range(1, len(segments)):
            prev = segments[i-1]
            curr = segments[i]

            m_condition = abs(curr["slope"]) < 0.5 / 73.5

            mean_condition = (
                prev["std"] < mean_std * 2 and
                curr["std"] < mean_std * 1.5 and
                (
                    curr["mean"] > prev["mean"] * 1.1 * 1.109435 or
                    curr["mean"] < prev["mean"] * 0.9 * 1.109435
                )
            )

            final = m_condition and mean_condition

            analyse_rows.append({
                "prev_start": prev["start"],
                "curr_start": curr["start"],
                "m": curr["slope"],
                "std_prev": prev["std"],
                "std_curr": curr["std"],
                "mean_prev": prev["mean"],
                "mean_curr": curr["mean"],
                "final": final
            })

            if final:
                new_starts.append(curr["start"])

        new_starts.append(len(values))

        # === Step 3: Shiften ===
        for i in range(1, len(new_starts)):
            s = new_starts[i-1]
            e = new_starts[i]

            segment_df = fit_curve.iloc[s:e].copy()

            shifted, shift = neumann_cam_shift(main_curve, segment_df)

            data.cuts.cuts.append({
                "start_date": segment_df.iloc[0]["Date"],
                "end_date": segment_df.iloc[-1]["Date"],
                "shift": shift
            })

            main_curve = pd.concat([main_curve, shifted], ignore_index=True)

        main_curve.sort_values(by='Date', inplace=True)

    if analyse:
        return pd.DataFrame(analyse_rows), pd.DataFrame(analyse_conditions)

    return main_curve


def start(data, analyse = False):
    start_time = time.time()
    data.data = pre_cleaning(data.data) # erstellt datum und JD, index separate 
    after_precleaning_time = time.time()
    data.normalize(normalize = True) # normalize
    after_normalize_time = time.time()
    ### ========== PREPROCESSING ALGORITHMS ========== ###
    data = remove_outliers_mad(data) # removes points around each camera
    after_remove_outliers_time = time.time()
    data.data = shift_cam_and_filters(data,analyse) #braucht am längsten
    after_shift_cam_time = time.time()
    #print(f"Cuts:\n{data.cuts.cuts}\noutliers:{data.cuts.outliers}\ncases:{data.cuts.cases}")
    if analyse:
        return data.data
    ### ================================ ###
        
    data.normalize(normalize = False) # back to orignal scale
    after_end_time = time.time()
    # print(f"\nTime Overview:"
    #     f"\nPrecleaning: {after_precleaning_time - start_time:.3f}s"
    #     f"\nNormalize: {after_normalize_time - after_precleaning_time:.3f}s"
    #     f"\nRemove Outliers: {after_remove_outliers_time - after_normalize_time:.3f}s"
    #     f"\nShift Cam/Filters: {after_shift_cam_time - after_remove_outliers_time:.3f}s"
    #     f"\nBack to Original Scale: {after_end_time - after_shift_cam_time:.3f}s"
    #     f"\nTotal: {after_end_time - start_time:.3f}s"
    # )    
    return data.data
