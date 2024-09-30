import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
# read the csv file
df = pd.read_csv('output.csv')
time = df['Time'].values
vehicles = df['NumberOfVehicles'].values
# interpolate, kind = 'cubic' or 'linear'
interp_function = interp1d(time, vehicles, kind='linear', fill_value="extrapolate")
# input # of hours ahead to 
future_times = np.linspace(min(time), 15, num=100)

predicted_vehicles = interp_function(future_times)

predicted_df = pd.DataFrame({'Time': future_times, 'PredictedVehicles': predicted_vehicles})
predicted_df.to_csv('predicted_vehicles.csv', index=False)

with open("predicted_vehicles.csv", "r") as f:
    for line in f: pass
    print(line) #this is the last line of the file