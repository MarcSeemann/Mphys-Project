# import lecroyparser
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import boost_histogram as bh
# from matplotlib import colors
# import csv


# # Functions

# def bin_pos_func(x_min, x_max, n_bins, p_per_bin):
#     bin_width = p_per_bin * 2
#     bin_loc_arr = np.linspace(x_min + bin_width / 2, x_max - bin_width / 2, n_bins)
#     return bin_loc_arr

# def number_of_bins_func(x, p_per_bin):
#     return (len(x) - len(x) % p_per_bin) // p_per_bin

# def get_points_per_bin(num_bins, num_points):
#     points_per_bin = num_points // num_bins
#     remainder = num_points % num_bins
#     if remainder > 0:
#         points_per_bin += 1
#     return points_per_bin


# def binned_data(y, n_bins, p_per_bin):
#     y = y[:n_bins * p_per_bin]
#     return np.sum(y.reshape(-1, p_per_bin), axis=1)


# def get_bin_centers(bin_edges):
#     # calculate the bin widths
#     bin_widths = np.diff(bin_edges)

#     # calculate the bin centers
#     bin_centers = bin_edges[:-1] + bin_widths / 2

#     return bin_centers




# x_arr = []
# y_arr = []

# # df = pd.DataFrame(x_arr)
# # df.columns = ['First']
# # df['Second'] = y_arr

# # print(df)

# count = 0

# # for i in range(10875):
# for i in range(200):
#     path = "E:\Manchester_Uni\Physics_Year_4\MphysProject\Hecker\Calibration\C1--PMT-test_calibration_long--" + str(str(i).zfill(5)) + ".trc"
#     data = lecroyparser.ScopeData(path)
#     if i == 0:
#         x_arr.append(data.x)
#         x_arr = x_arr[0]*1000000
#         points_per_bin = 100
#         n_bins = number_of_bins_func(x_arr, points_per_bin)
#         hist_val_x, bin_edges_x = np.histogram(x_arr, bins=n_bins)
#         bin_centres_x = get_bin_centers(bin_edges_x)

#         # Dataframe creation and storing

#         df = pd.DataFrame(bin_centres_x)
#         df.columns = ['Time']
#         df['Amp 0'] = binned_data(data.y*-1, n_bins, points_per_bin)
#         df.to_hdf('test.h5', key='mydata', mode='w')

#     else:
#         df = pd.read_hdf('test.h5')
#         df['Amp' + str(i)] = binned_data(data.y*-1, n_bins, points_per_bin)
#         df.to_hdf('test.h5', key='mydata', index=False)
#     #     df.to_csv('test.csv', mode='w')

#     # else:
#     #     df = pd.read_csv('test.csv')
#     #     df['Amp' + str(i)] = binned_data(data.y*-1, n_bins, points_per_bin)
#     #     df.to_csv('test.csv', index=False)

#     if(count % 5 == 0):
#         print(count)
#     count += 1


def find_slope(x1, y1, x2, y2):
    """
    Finds the slope between two points given their coordinates.
    
    Parameters:
    point1 (tuple): The coordinates of the first point as a tuple of (x, y) values.
    point2 (tuple): The coordinates of the second point as a tuple of (x, y) values.
    
    Returns:
    float: The slope of the line passing through the two points.
    """
    # unpack the coordinates of the two points
    
    # calculate the slope using the slope formula

    
    return (y2 - y1) / (x2 - x1)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

df = pd.read_hdf('combined_calib_test.h5')
col_A = df['Amp 0'].values

x_arr = df['Time'].values
y_arr = []

for column in df.columns[1:]:
    # extract the data of the column as a numpy array and append it to the list
    y_arr.append(df[column].values)

event = 5
temp_var = 3560

peaks, _ = find_peaks(y_arr[event], height=0.2)
plt.scatter(x_arr,y_arr[event], s=4, color='blue')
plt.plot(x_arr[peaks], y_arr[event][peaks], "y")
plt.xlabel("Sample Time (ns)", fontsize = 25)
plt.ylabel("ADC Value", fontsize = 25)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)


plt.scatter(x_arr[temp_var], y_arr[event][temp_var], color='red', s=4)
plt.legend()
plt.show()

print(find_slope(x_arr[3558], y_arr[event][3558], x_arr[3559], y_arr[event][3559]))


