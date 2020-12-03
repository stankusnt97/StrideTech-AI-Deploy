import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("initial_data_cleaned.csv")

data = pd.read_csv("initial_data_cleaned.csv")

plt.scatter(data['time_stamp'], data['left_fsr_reading_mv'])

data.shape
data.head()

data['Activity'].value_counts()

# data processed at 5 times/second
Fs = 50
activities = data['Activity'].value_counts().index
activities

def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15,7), sharex = False)
    plot_axis(ax0, data['time_stamp'], data['left_fsr_reading_mv'], 'Left FSR Reading (mV)')
    plot_axis(ax1, data['time_stamp'], data['right_fsr_reading_mv'], 'Right FSR Reading (mV)')
    plot_axis(ax2, data['time_stamp'], data['hip_distance'], 'Hip Distance (in)')
    fig.suptitle(activity)
    plt.show()

def plot_axis(ax, x, y, title):
    ax.plot(x, y, 'g')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

for activity in activities:
    data_for_plot = data[(data['Activity'] == activity)][:Fs*5]
    plot_activity(activity, data_for_plot)