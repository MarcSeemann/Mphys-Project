# %%
import lecroyparser
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import boost_histogram as bh
from matplotlib import colors
import csv
import time
import os


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

START_ITER = 0
LENGTH_WHILE = 1000

df = {}

for i in range(1):
    x_arr = []
    y_arr = []


    # %%
    count = 0

    # def foo2_dict(df):
    #     new_columns = pd.DataFrame({f"Amp{i}": binned_data(data.y*-1, n_bins, points_per_bin) for i in range(10000)})
    #     return pd.concat([df, new_columns], axis=1)


    count_in_loop = 0 + LENGTH_WHILE * i + START_ITER
    error_counter = 0 

    while (count_in_loop < LENGTH_WHILE * (i+1)):
        path = "F:\Test_waveforms\C1--PMT-test_calibration_long--" + str(str(count_in_loop).zfill(5)) + ".trc"
        try:
            data = lecroyparser.ScopeData(path)
        except FileNotFoundError:
            # time.sleep(1)
            error_counter += 1
            if error_counter == 3:
                print('File not found, skipped')
                count_in_loop += 1
            continue
        error_counter = 0

        if count_in_loop == 0 + LENGTH_WHILE * i + START_ITER:
            x_arr.append(data.x)
            x_arr = x_arr[0]*1000000
            points_per_bin = 100
            n_bins = number_of_bins_func(x_arr, points_per_bin)
            hist_val_x, bin_edges_x = np.histogram(x_arr, bins=n_bins)
            bin_centres_x = get_bin_centers(bin_edges_x)

            # Dataframe creation and storing

            df[i] = pd.DataFrame(bin_centres_x)
            df[i].columns = ['Time']
            df[i]['Amp 0'] = binned_data(data.y*-1, n_bins, points_per_bin)
            # df.to_hdf('test.h5', key='mydata', mode='w')

        else:
            # df = pd.read_hdf('test.h5')
            new_column = pd.DataFrame({"Amp " +str(count_in_loop) : binned_data(data.y*-1, n_bins, points_per_bin)})
            df[i] = pd.concat([df[i], new_column], axis=1)

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

        if(count_in_loop % 100 == 0):
            print(count_in_loop)
        count_in_loop += 1




    # %%
    df[i].to_hdf('extra_test' + str(i) + '.h5', key='mydata', index=False)

    # %%
    # df = pd.read_hdf('test.h5')
    # print(df)
    # second_loop_count = 0 + LENGTH_WHILE * i
    # while (second_loop_count < LENGTH_WHILE * (i+1)):
    #     try:
    #         os.remove("E:\Manchester_Uni\Physics_Year_4\MphysProject\Hecker\Calibration\C1--PMT-test_calibration_long--" + str(str(second_loop_count).zfill(5)) + ".trc")
    #     except FileNotFoundError:
    #         second_loop_count += 1
    #         continue
    #     second_loop_count += 1
        
        


