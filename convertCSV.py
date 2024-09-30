import pandas as pd
#read csv
df = pd.read_csv('information.csv')
# create function time_to_float
def time_to_float(time_str):
    # split by :
    time_parts = time_str.split(':')
    hours = int(time_parts[0])
    # divide by 60 to translate into floating point
    minutes = int(time_parts[1]) / 60
    return round(hours + minutes, 2)
df['Time'] = df['Time'].apply(time_to_float)
# delete date column (irrelevant rn)
df = df.drop('Date', axis=1)
df.to_csv('output.csv', index=False)
