import lecroyparser
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import boost_histogram as bh
from matplotlib import colors


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
count = 0
# for i in range(10875):
for i in range(100):
    path = "F:\Test_waveforms\C1--PMT-test_calibration_long--" + str(str(i).zfill(5)) + ".trc"
    data = lecroyparser.ScopeData(path)
    if i == 0:
        x_arr.append(data.x)
        
        x_arr = x_arr[0]*1000000
        points_per_bin = 100
        n_bins = number_of_bins_func(x_arr, points_per_bin)
        hist_val_x, bin_edges_x = np.histogram(x_arr, bins=n_bins)
        bin_centres_x = get_bin_centers(bin_edges_x)

    y_arr.append(data.y*-1)

    for i in range(len(y_arr)):
        binned_data(y_arr[i], n_bins, points_per_bin)
    if(count % 20 == 0):
        print(count)
    count += 1

    




# Unpack x value list into single numpy arr
x_arr = x_arr[0]*1000000
points_per_bin = 100
n_bins = number_of_bins_func(x_arr, points_per_bin)

binned_y = []

# binned_x = np.linspace(np.min(x_arr), np.max(x_arr), number_of_bins_func(x_arr, points_per_bin))


hist_val_x, bin_edges_x = np.histogram(x_arr, bins=n_bins)
bin_centres_x = get_bin_centers(bin_edges_x)

print(len(bin_centres_x))

for i in range(len(y_arr)):
    
    binned_y.append(binned_data(y_arr[i], n_bins, points_per_bin))
plt.plot(x_arr, y_arr[0])
plt.show()
plt.plot(bin_centres_x, binned_y[0])
plt.show()


transposed_array_y = list(map(list, zip(*binned_y)))


with open('new_test_calib.txt', 'w') as file:
    # # Write each element of x_arr to a new line in the file
    # for x in bin_centres_x:
    #     file.write(str(x) + "\n")
    # # Write an empty line to separate the two arrays
    # file.write("\n")
    # # Write each row of the transposed array to the file
    # for row in transposed_array_y:
    #     row_str = ", ".join(str(x) for x in row)
    #     file.write(row_str + "\n")
    for i, row in enumerate(transposed_array_y):
        row_str = ", ".join(str(x) for x in [bin_centres_x[i]] + row)
        file.write(row_str + "\n")


# # Number of bins
# n_bins = 100

# # Create empty list to store binned numpy arrays

# binned_list_y = []
# bin_edges_list_y = []

# hist_val_x, bin_edges_x = np.histogram(x_arr, bins=n_bins)

# for arr in y_arr:
#     hist, bin_edges = np.histogram(arr, bins=n_bins)
#     binned_list_y.append(np.array(hist))
#     bin_edges_list_y.append(bin_edges)

# # Loop through each numpy array in the list

# for arr in y_arr:
#     # Use pandas cut() function to bin the data
#     bins, bin_edges = pd.cut(arr, bins=n_bins, retbins=True)
#     # Convert binned data to numpy array and append to binned_list
#     binned_list_y.append(np.array(bins.value_counts()))
#     bin_edges_list_y.append(bin_edges)








# # Setting constants for plot

# n_bins = 200
# x_range_min = np.min(x_arr)
# x_range_max = np.max(x_arr)
# y_range_min = np.min(y_arr[0])
# y_range_max = np.max(y_arr[0])


# h = bh.Histogram(bh.axis.Regular(n_bins, x_range_min, x_range_max, metadata="Momentum, p [GeV]")
#                  , bh.axis.Regular(n_bins, y_range_min, y_range_max, metadata="Track time [ns]"))

# h.fill(x_arr, y_arr[0])
# w_1, x_binned_1, y_binned_1 = h.to_numpy()
# print(min(x_binned_1))
# print(max(x_binned_1))

# # Plotting histogram

# fig, ax = plt.subplots()
# fig.set_dpi(300)
# mesh = ax.pcolormesh(x_binned_1, y_binned_1, w_1.T, norm=colors.LogNorm(vmin=10**(-2), vmax=10**3), cmap='winter')
# ax.set_title('Helium Track time vs Momentum')
# ax.set_xlabel(h.axes[0].metadata)
# ax.set_ylabel(h.axes[1].metadata)
# fig.colorbar(mesh)
# plt.show()


