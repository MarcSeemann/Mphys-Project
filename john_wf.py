import uproot
import matplotlib.pyplot as plt
import numpy as np
import boost_histogram as bh
from numba import jit


print("Trying to open files ...")

# with uproot.open('Run203-PMT107.root') as file:
#     events_simulation = file
#     for i in file['Tree;4/']:
#         print(i)
#     # data_res = events_simulation['Long_tuple/DecayTree;1/' + residual_p].array(library='np')
#     if not file.closed:
#         file.close()


# # Viewing particular event of particular file.
# # Open the data, apply to variable

file = "E:\Manchester_Uni\Physics_Year_4\MphysProject\Hecker\Run103-PMT107.root"
# file = "E:\Manchester_Uni\Physics_Year_4\MphysProject\Hecker\Boulby_107_Signal.root"

tree = uproot.open(file)["Tree;3"]
branches = tree.arrays()

# how long between data taken
timegate = 2
# length of event
eventno = len(branches['ADC'][0])
time = []
# Creating list for sample times that are 2ns intervals, 150 samples
for i in range(eventno):
    time.append(i*timegate)


# total_n_events = len(branches['ADC'])
total_n_events = 1000

print(total_n_events)

time_list = []

time_list = time * total_n_events

@jit
def sum_list(n_events):
    list_v = []
    for i in range(n_events):
        for j in range(eventno):
            list_v.append(branches['ADC'][i][j])
    return list_v



sum_list = sum_list(total_n_events)


# Setting constants for plot

n_bins = 500
x_range_min = min(time)
x_range_max = max(time)
y_range_min = min(sum_list)
y_range_max = max(sum_list)


h = bh.Histogram(bh.axis.Regular(n_bins, x_range_min, x_range_max, metadata="Time")
                 , bh.axis.Regular(n_bins, y_range_min, y_range_max, metadata="ADC Value"))

h.fill(time_list, sum_list)
w, x_binned_1, y_binned_1 = h.to_numpy()


# Plotting histogram

fig, ax = plt.subplots()
fig.set_dpi(300)
mesh = ax.pcolormesh(x_binned_1, y_binned_1, w.T, cmap='nipy_spectral_r')
ax.set_title('2D Histogram of ADC vs time')
ax.set_xlabel(h.axes[0].metadata)
ax.set_ylabel(h.axes[1].metadata)
fig.colorbar(mesh)
plt.show()




# # Input event
# Nevent = int(input("What event do you wish to view? "))

# plt.plot(time,branches['ADC'][Nevent])
# plt.xlabel("Sample Time (ns)", fontsize = 17)
# plt.ylabel("ADC Value", fontsize = 17)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.title("Event " + str(Nevent), fontsize = 22)
# plt.show()



# Loop through entire data and find mean ADC value of each +- 20 and if any points outside this
# range then identify as a peak.
# Check the bachelor thesis on how to create the histograms.


# while(True):
    


# counter = 8400
# while(True):
#     # Input event
#     # Nevent = int(input("What event do you wish to view? "))
#     Nevent = counter

#     plt.plot(time,branches['ADC'][Nevent])
#     plt.xlabel("Sample Time (ns)", fontsize = 17)
#     plt.ylabel("ADC Value", fontsize = 17)
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     plt.title("Event " + str(Nevent), fontsize = 22)
#     plt.show()
#     counter += 1
