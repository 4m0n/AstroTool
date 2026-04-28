from pathlib import Path
from tqdm import tqdm
import config
from pathlib import Path
from scripts.base import LightCurve
from scripts.base import Parameters
import scripts.base as base
from astropy.time import Time
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import AutoDateLocator
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scripts.evaluate as evaluate
value = config.VALUE_CALCULATION

farben = ["blue", "purple", "green", "orange", "red",
    "cyan", "magenta", "black", "brown", "darkblue", "lime",
    "indigo", "gold", "darkgreen", "teal", "red",
    "maroon", "navy", "darkred", "forestgreen",
    "slategray", "darkslateblue", "chocolate", "darkorange",
    "seagreen", "sienna", "darkmagenta", "midnightblue",
    "firebrick", "cadetblue", "dodgerblue", "peru",
    "rosybrown", "saddlebrown", "darkolivegreen", "steelblue",
    "tomato", "mediumblue", "deepskyblue", "crimson", "mediumvioletred",
    "orchid", "plum", "slateblue", "turquoise", "violet", "darkcyan",
    "darkorchid", "mediumorchid"]


def normalize(data,shift=None,scale = None):
    if shift == None:
        shift = data[value].min()
        data[value] = data[value] - shift
        scale = data[value].max()
        data[value] = data[value] / scale
        return data,scale,shift
    else:
        data[value] = data[value] * scale
        data[value] = data[value] + shift
        return data
    
def add_filter_to_cams(file):
    for i in range(len(file)):
        file.loc[i, "Filter_Camera"] = f'{file.loc[i, "Filter"]}-{file.loc[i, "Camera"]}'
    return file

def print_that_shit(string):
    for idx, row in string.iterrows():
        print("\n" + "="*79)
        print(
            f"Eintrag {idx} | Bedingung 1 {bool(row['bedingung'])} "
            f"Bedingung 2 {bool(row['bedingung2'])} m {bool(row['m'])} "
            f"| Camera: {row['Camera']}"
        )
        print("-"*79)

        # --- Werte aus DataFrame holen ---
        m_flag   = bool(row["m"])
        m_value  = float(row["m_value"])

        std_prev  = float(row["cs[std][k-1]"])
        std_curr  = float(row["cs[cut][k]"])          # hier steckt dein "std danach" drin
        mean_prev = float(row["cs[mean][k-1]"])
        mean_curr = float(row["cs[mean][k]"])
        mean_std  = float(row["mean_std"])
        ratio     = float(row["mean[k]/mean[k-1]"])

        bed1_flag = bool(row["bedingung"])
        bed2_flag = bool(row["bedingung2"])

        # --- Bedingungen nachrechnen (ohne timediff) ---

        # m
        m_recalc = abs(m_value) < 0.5

        # Bedingung 1:
        # (std_prev < mean_std*1.2) and (std_curr < mean_std*1.1)
        # and (mean_curr > mean_prev*1.05 or mean_curr < mean_prev*0.95)
        b1_a  = std_prev < mean_std * 1.2
        b1_b  = std_curr < mean_std * 1.1
        b1_c1 = mean_curr > mean_prev * 1.05
        b1_c2 = mean_curr < mean_prev * 0.95
        b1_c  = b1_c1 or b1_c2
        bed1_recalc = b1_a and b1_b and b1_c

        # Bedingung 2 (nur Ratio-Teil, ohne timediff):
        # (abs(ratio) > 2) or (abs(ratio) < 0.5)
        ratio_abs = abs(ratio)
        b2_r1 = ratio_abs > 2.0
        b2_r2 = ratio_abs < 0.5
        bed2_recalc = b2_r1 or b2_r2

        # ======================
        # Bedingung m
        # ======================
        print("Bedingung m:")
        print("  m = abs(m_value) < 0.5")
        print(f"    abs({m_value:.5f}) < 0.5")
        print(f"    Ergebnis: m_recalc={m_recalc}, gespeichert m={m_flag}")

        # ======================
        # Bedingung 1
        # ======================
        print("\nBedingung 1 (std & mean-Sprung):")
        print(
            "  (std_prev < mean_std*1.2) AND (std_curr < mean_std*1.1) "
            "AND (mean_curr > mean_prev*1.05 OR mean_curr < mean_prev*0.95)"
        )
        print(
            f"    ({std_prev:.5f} < {mean_std*1.2:.5f}) AND "
            f"({std_curr:.5f} < {mean_std*1.1:.5f}) AND "
            f"({mean_curr:.5f} > {mean_prev*1.05:.5f} OR "
            f"{mean_curr:.5f} < {mean_prev*0.95:.5f})"
        )
        print(
            "    Ergebnis: "
            f"a={b1_a}, b={b1_b}, c1={b1_c1}, c2={b1_c2}, "
            f"(c1 OR c2)={b1_c}, gesamt={bed1_recalc}, gespeichert bedingung={bed1_flag}"
        )

        # ======================
        # Bedingung 2
        # ======================
        print("\nBedingung 2 (Ratio-Sprung):")
        print("  (abs(mean[k]/mean[k-1]) > 2.0) OR (abs(mean[k]/mean[k-1]) < 0.5)")
        print(
            f"    abs({ratio:.5f}) > 2.0 OR abs({ratio:.5f}) < 0.5"
        )
        print(
            "    Ergebnis: "
            f"r1(abs(ratio)>2)={b2_r1}, r2(abs(ratio)<0.5)={b2_r2}, "
            f"gesamt={bed2_recalc}, gespeichert bedingung2={bed2_flag}"
        )

def get_cut_intervals(df,ax, data):
    cuts, cuts_data = data.preprocess(True)
    cam = df["Camera"].copy()
    cameras = cam.unique()
    for i, cut in cuts_data.iterrows():
        color = farben[np.where(cameras == cut["Camera"])[0][0]]
        ax.vlines(cut["Date"], ymin=df[value].min(), ymax=df[value].max(), color = color)

def get_param_list(param = None, threshold=0.5):
    data = base.load_processed_data(None, False)
    df = pd.DataFrame(columns=["name", "fvar"])
    rows = []
    for val in data:
        name = val.original_name.removesuffix("-light-curves")
        fvar = val.parameters.Fvar
        rows.append({"ID": name, "fvar": fvar})
        df = pd.concat([df, pd.DataFrame([{"ID": name, "fvar": fvar, "length":val.new_length}])], ignore_index=True)

    df = df[df["fvar"]>= threshold]
    
    ref = pd.read_csv("../references/name_id.csv")
    df.reset_index(inplace=True)
    for i in range(len(df)):
        for k in range(len(ref)):
            if int(df.iloc[i]["ID"]) == int(ref.iloc[k]["ID"]):
                df.at[i, "name"] = ref.iloc[k]["name"]
                data = base.LightCurve.load(df.iloc[i]["ID"])
                data.name = ref.iloc[k]["name"]
                data.save()
                break
    
    df.sort_values("fvar",inplace=True, ascending=False)
    df.reset_index(inplace=True)
    df = df.rename(columns={"ID": "ASAS-SN-ID"})
    df.drop(["level_0", "index"], axis=1, inplace=True)
    df["name"] = df["name"].fillna("unknown")
    df.to_csv("Fvar_list2.csv", index=False)
    return df
    
def create_parameter_list():
    data = base.load_processed_data(None, False)
    df = pd.DataFrame()
    for val in data:
        params = val.parameters
        entry = {}
        for name, value in params.__dict__.items():
            if name == "frequency":
                entry[name] = value["frequency"].values[0] if value["frequency"].values.size > 0 else None
                entry["frequency_strength"] = value["properties"].values[0] if value["properties"].values.size > 0 else None
            else:
                entry[name] = value
        entry["name"] = val.get_name()
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        #df.drop(["frequency"], axis=1, inplace=True)
    df.to_csv("parameters_overview.csv")
    return df

def compare_before_after_preprocessing(data,title = None):
    #before, after = data.load_orignal_thesis_data(keep = False), data.load_processed_data()
    before, after = data.data, data.load_processed_data()

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    
    def plot_file(file, label_suffix, ax):
        if label_suffix == "before":
            ...
            #error_data = get_cut_intervals(file,ax,data)
        cam = file["Camera"].copy()
        cameras = cam.unique()
        c3, c4, c_general = [], [], []
        for index, i in enumerate(cam):
            try:
                color = farben[np.where(cameras == i)[0][0]]
            except:
                color = "black"
            if file["Filter"].iloc[index] == "V":
                c3.append(color)
            else:
                c4.append(color)
            c_general.append(color)
        x_1 = file.loc[file["Filter"] == "V","Date"].values.copy()
        x_2 = file.loc[file["Filter"] == "g","Date"].values.copy()
        y_1 = file.loc[file["Filter"] == "V", value].values.copy()
        y_2 = file.loc[file["Filter"] == "g", value].values.copy()

        pointsize = 18
        
            
        ax.scatter(x_1, y_1, c=c3, alpha=0.6, zorder=5, marker="x", s=pointsize, label=f"V-Band {label_suffix}")
        ax.scatter(x_2, y_2, c=c4, alpha=0.6, zorder=5, marker="o", s=pointsize, label=f"g-Band {label_suffix}")
        

        return x_1, x_2

    x1_1, x1_2 = plot_file(before, "before", axs[0])
    x2_1, x2_2 = plot_file(after, "after", axs[1])

    text_size1 = 18
    text_size2 = text_size1 - 3

    for ax, label in zip(axs, ["Raw", "Processed"]):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
        ax.xaxis.set_major_locator(AutoDateLocator(maxticks=5))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', linewidth=0.4, alpha=0.2, color='black')
        ax.grid(which='major', linestyle='--', linewidth=0.4, alpha=0.3, color='black')
        ax.set_xlabel("Date", fontsize=text_size1)
        ax.set_ylabel(r"Flux [mJy]", fontsize=text_size1)
        ax.set_title(label, fontsize=text_size1)
        legend = ax.legend(fontsize=text_size2, frameon=True, fancybox=True, framealpha=0.7)
        for handle in legend.legend_handles:
            handle.set_facecolor('black')
        ax.tick_params(axis='x', labelsize=text_size2)
        ax.tick_params(axis='y', labelsize=text_size2)
        ax.yaxis.get_offset_text().set_fontsize(text_size2)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)

    # xlim anpassen
    x_all_before = np.concatenate([x1_1, x1_2])
    x_all_after = np.concatenate([x2_1, x2_2])
    if title == None:
        fig.suptitle(data.get_name(), fontsize=text_size1 + 2)
    else:
        fig.suptitle(title, fontsize=text_size1 + 2)
    axs[0].set_xlim(min(x_all_before) - pd.Timedelta(days=100), max(x_all_before) + pd.Timedelta(days=100))
    axs[1].set_xlim(min(x_all_after) - pd.Timedelta(days=100), max(x_all_after) + pd.Timedelta(days=100))

    plt.show()   
    
def frequency_plot():
    """
    Plot aus allen Objekten mit 3 stärksten frequenzen frequency vs strength dargestellt
    """
    collection = base.load_processed_data(None, False)
    frequncy = pd.DataFrame()
    length = []
    for data in collection:
        frequncy = pd.concat([frequncy, data.parameters.frequency[["time","properties"]]], ignore_index=True)
        length.append(max(data.data["JD"])-min(data.data["JD"]))
    plt.scatter(frequncy["time"], frequncy["properties"], alpha=0.6)
    plt.vlines(x=length, ymin=min(frequncy["properties"]), ymax=0.15, colors="red", alpha = 0.2)
    plt.xlabel("Period duration T [days]")
    plt.ylabel("Strength")
    plt.show()
    
def threshold_plot(lower = None, upper = None, parameter = None):
    collection = base.load_processed_data(None, False)
    for data in collection:
        try:
            param = getattr(data.parameters, parameter)
        except:
            raise ValueError(f"Parameter {parameter} not found.")
        if type(param) == type(pd.DataFrame()):
            if len(param) == 0:
                continue
            
        if parameter == "frequency":
            compare_value = float(param["time"].iloc[0])
            if param['properties'].iloc[0] < 0.3:
                continue
                pass
        else:
            compare_value = float(param)
        if lower == None:
            lower = -np.inf
        if upper == None:
            upper = np.inf    
        if compare_value >= lower and compare_value <= upper:
            if parameter == "frequency":
                compare_before_after_preprocessing(data,f"Name: {data.get_name()} \n{parameter}: {round(compare_value/365,1)} strength: {round(param['properties'].iloc[0],3)}")
            else:
                compare_before_after_preprocessing(data,f"Name: {data.get_name()} \n{parameter}: {round(compare_value,3)}")
        
    
def parameter_distribution_plot(parameter = None, bins=80):
    
    def hist_plot(values, param_name, bins):
        values = np.array(values)
        mean = np.mean(values)
        median = np.median(values)
        sns.set(style="whitegrid", palette="muted", font_scale=1.2)

        plt.figure(figsize=(12, 7))

        #plt.xscale('log')
        sns.histplot(values, bins=bins, kde=True, color="#3498db", edgecolor="black", alpha=0.7, log_scale=False)
        # try:
        #     print(f"VALUES {param_name}: max: {max(values)} min: {min(values)} \n{values}")
        # except:
        #     ...
        #plt.axvline(mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean:.2f}")
        #plt.axvline(median, color="green", linestyle=":", linewidth=2, label=f"Median: {median:.2f}")
        if param_name == "frequency":
            plt.xlabel("T in days")
            plt.title(f"Distribution of T (n={len(values)})", fontweight="bold")
        else:
            plt.xlabel(param_name)
            plt.title(f"Distribution of {param_name} (n={len(values)})", fontweight="bold")
        plt.ylabel("Count")
        #plt.legend()
        plt.tight_layout()
        plt.show()
    
    data = base.load_processed_data(None, False)
    
    parameters_class = Parameters()
    variablen_namen = list(parameters_class.__dict__.keys())
    print(f"variablen_namen: {variablen_namen}")

    parameter_list = {}

    # Index (Lichtkurvennamen) einmal festlegen
    curve_names = [obj.original_name for obj in data]

    for param_name in variablen_namen:
        if parameter is not None and param_name not in parameter:
            continue

        values = []

        if param_name == "Fvar":
            for obj in data:
                param = getattr(obj.parameters, param_name, None)
                values.append(param if param is not None else np.nan)

            hist_plot(values, param_name, bins)

        elif param_name == "frequency":
            for obj in data:
                param = getattr(obj.parameters, param_name, None)

                if param is not None and len(param) > 0:
                    val = param["frequency"].iloc[0]
                    values.append(1 / val if val is not None else np.nan)
                else:
                    values.append(np.nan)

            hist_plot(values, param_name, bins)

        else:
            for obj in data:
                param = getattr(obj.parameters, param_name, None)
                values.append(param if param is not None else np.nan)

            hist_plot(values, param_name, bins)

        parameter_list[param_name] = values

    # DataFrame korrekt erstellen
    df = pd.DataFrame(parameter_list, index=curve_names)
    df.to_csv(config.STATISTICS_DIR / "parameter_distribution.csv")
    print(df)
    
def sorted_parameters(data = None, parameters_df = None): # aufrufen um lichtkurven nach parameter sortiert absteigend zu plotten
    if data == None:
        data = base.load_processed_data(None, False)
    if parameters_df == None:
        parameters_df = pd.read_csv(config.STATISTICS_DIR / "parameter_distribution.csv", index_col=0)
    param_columns = parameters_df.columns.values
    print_string = "Press: "
    for idx, val in enumerate(param_columns):
        print_string += f"{idx}: {val} | "
    print_string + "Input: "    
    user_input = int(input(print_string))
    
    parameters_df.sort_values(by=param_columns[user_input], ascending=False, inplace=True)
    
    for name in parameters_df.index:
        for obj in data:
            if obj.original_name == name:
                compare_before_after_preprocessing(obj,f"Name: {obj.get_name()} \n{param_columns[user_input]}: {round(getattr(obj.parameters, param_columns[user_input]),3)}")
                break
    
    
if __name__ == "__main__":
    
    
    
    if False:
        #create_parameter_list()
        parameter_distribution_plot()
    
    if False: # asas_sn test
        objs = base.LightCurve.load_asas_sn()
        for obj in objs:
            compare_before_after_preprocessing(obj)
        
    if False:    
        #df = get_param_list("Fvar", 0.3)
        liste = df["ASAS-SN-ID"].values
        for val in liste:
            data = base.LightCurve.load(val)
            compare_before_after_preprocessing(data)
    
    if False: # load single
        data = LightCurve.load("549756880980") 
        
        print(f"Data: \n{data.data}")
        
        compare_before_after_preprocessing(data)
    if True:
        sorted_parameters()
        
"""
- Fvar kann negativ -> nicht berechnebar sein
- frequenz top 1 oder top 3 frequenzen

####### noch machen
- bei evaluation für R nicht normalisieren
- plot bei parameter ausgabe schöner machen


Problematische kurven: 
- 515396305126-light-curves
- 549756880980-light-curves
"""


