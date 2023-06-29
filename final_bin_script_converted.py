# %%
import lecroyparser
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import boost_histogram as bh
from matplotlib import colors
import csv


# Functions

def bin_pos_func(x_min, x_max, n_bins, p_per_bin):
    bin_width = p_per_bin * 2
    bin_loc_arr = np.linspace(x_min + bin_width / 2, x_max - bin_width / 2, n_bins)
    return bin_loc_arr

def number_of_bins_func(x, p_per_bin):
    return (len(x) - len(x) % p_per_bin) // p_per_bin

def get_points_per_bin(num_bins, num_points):
    points_per_bin = num_points // num_bins
    remainder = num_points % num_bins
    if remainder > 0:
        points_per_bin += 1
    return points_per_bin


def binned_data(y, n_bins, p_per_bin):
    y = y[:n_bins * p_per_bin]
    return np.sum(y.reshape(-1, p_per_bin), axis=1)


def get_bin_centers(bin_edges):
    # calculate the bin widths
    bin_widths = np.diff(bin_edges)

    # calculate the bin centers
    bin_centers = bin_edges[:-1] + bin_widths / 2

    return bin_centers




x_arr = []
y_arr = []

# df = pd.DataFrame(x_arr)
# df.columns = ['First']
# df['Second'] = y_arr

# print(df)




# %%
count = 0

def foo2_dict(df):
    new_columns = pd.DataFrame({f"Amp{i}": binned_data(data.y*-1, n_bins, points_per_bin) for i in range(10000)})
    return pd.concat([df, new_columns], axis=1)


for i in range(9947):
# for i in range(5):
    path = "F:\Test_waveforms\C1--PMT-test_calibration_long--" + str(str(i).zfill(5)) + ".trc"
    data = lecroyparser.ScopeData(path)
    if i == 0:
        x_arr.append(data.x)
        x_arr = x_arr[0]*1000000
        points_per_bin = 150
        n_bins = number_of_bins_func(x_arr, points_per_bin)
        hist_val_x, bin_edges_x = np.histogram(x_arr, bins=n_bins)
        bin_centres_x = get_bin_centers(bin_edges_x)
        # Dataframe creation and storing

        df = pd.DataFrame(bin_centres_x)
        df.columns = ['Time']
        df['Amp 0'] = binned_data(data.y*-1, n_bins, points_per_bin)
        # df.to_hdf('test.h5', key='mydata', mode='w')

    else:
        # df = pd.read_hdf('test.h5')
        new_column = pd.DataFrame({"Amp " +str(i) : binned_data(data.y*-1, n_bins, points_per_bin)})
        df = pd.concat([df, new_column], axis=1)

        # df = pd.DataFrame(bin_centres_x)
        # df.columns = ['Time']
        # df['Amp' + str(i)] = binned_data(data.y*-1, n_bins, points_per_bin)


        # foo2_dict(df)
        # pd.concat(binned_data(data.y*-1, n_bins, points_per_bin) ,axis=1)

        # df.to_hdf('test.h5', key='mydata', index=False)
    #     df.to_csv('test.csv', mode='w')

    # else:
    #     df = pd.read_csv('test.csv')
    #     df['Amp' + str(i)] = binned_data(data.y*-1, n_bins, points_per_bin)
    #     df.to_csv('test.csv', index=False)

    if(count % 20 == 0):
        print(count)
    count += 1




# %%
df.to_hdf('new_test.h5', key='mydata', index=False)

# %%
# df = pd.read_hdf('test.h5')
# print(df)


