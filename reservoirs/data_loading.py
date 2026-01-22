# Note: Fichier permettant l'import des données d'un fichier pour le traitement dans les fichier Ridge, Reservoir et Hyperopt 

import csv 
import os 
import math 
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt 

#Makes the time a continuous value to work with ESN.
def time_linearisation(date): 
    format_code = "%Y%m%d"
    dt = datetime.strptime(date, format_code)

    # Week day
    dow = dt.weekday()
    dow_sin = math.sin(2 * math.pi * dow / 7)
    dow_cos = math.cos(2 * math.pi * dow / 7)

    # Month day
    dom = dt.day - 1
    days_in_month = 31  # number of days in a month approximation
    dom_sin = math.sin(2 * math.pi * dom / days_in_month)
    dom_cos = math.cos(2 * math.pi * dom / days_in_month)

    # Day of Year
    doy = dt.timetuple().tm_yday - 1
    days_in_year = 365.25  # number of days in a year approximation
    doy_sin = math.sin(2 * math.pi * doy / days_in_year)
    doy_cos = math.cos(2 * math.pi * doy / days_in_year)

    # Months
    month = dt.month - 1
    month_sin = math.sin(2 * math.pi * month / 12)
    month_cos = math.cos(2 * math.pi * month / 12)

    return np.array([
        doy_sin, doy_cos,
    ])

# Load preprocessed data

# inputs : Qls (1), Qmmj (2), Ptot (9), Fsol(10), Temp (11), E_ou (12), E_PE (13), E_PM (14), Vent (15), Humi (16), Dli (17), SSI (18), IHGR (19), TN (22), TX (23), niveau_deau (24)

def load_X_Y(filepath):
    X = []
    inputs_columns = { 24 }
    with open(filepath, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not row or row[0].startswith("D"):
                continue
            time_features = time_linearisation(row[0])
            X.append(np.concatenate([time_features] + [[float(row[i])] for i in inputs_columns]))
        return np.array(X)

def plot_results(y_pred, y_test, sample=800, diff = False):
    #fig = plt.figure(figsize=(15, 7))
    plt.subplot(211)
    plt.plot(np.arange(sample), y_pred[:sample], lw=3, label="ESN prediction")
    plt.plot(np.arange(sample), y_test[:sample], linestyle="--", lw=2, label="True value")
    if diff:
        plt.plot(np.abs(y_test[:sample] - y_pred[:sample]), label="Absolute deviation")

    plt.legend()
    plt.show()