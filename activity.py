import numpy as np
import matplotlib.pyplot as plt
import os


def read_last_row(filename):
    '''Reads a text file with two numbers on each row separated by a comma,
    and returns the two numbers on the last row.
    Returns: Tuple (num1, num2)'''
    folderpath = r"E:\\Manchester_Uni\\Physics_Year_4\\MphysProject\\Hecker\\Activity"
    with open(os.path.join(folderpath,filename), 'r') as f:
        lines = f.readlines()
    
    last_line = lines[-1].strip()
    num1, num2 = last_line.split(',')
    num1 = float(num1)
    num2 = float(num2)
    
    return num1, num2


transmitted_arr = []
absorbed_arr = []



for i in range(96):
    file_n = str("particle_output_") + str(i) + str(".txt")
    trans, absorb = read_last_row(file_n)
    transmitted_arr.append(trans)
    absorbed_arr.append(absorb)


transmitted_arr = np.array(transmitted_arr)
absorbed_arr = np.array(absorbed_arr)

total_arr = absorbed_arr + 50000



# plt.hist(total_arr, bins=30)
# plt.xlabel("Activity (total decays)", fontsize = 17)
# plt.ylabel("Number of runs", fontsize = 17)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.title("Simulated activity of source", fontsize = 22)
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# textstr = str("Mean activity = ") + str()
# # place a text box in upper left in axes coords
# plt.text(0.05, 0.95, textstr, transform=plt.transAxes, fontsize=14,
#     verticalalignment='top', bbox=props)
# plt.legend()
# plt.show()



mean_activity = np.mean(total_arr)

fig, ax = plt.subplots()


ax.set_xlabel("Activity (total decays)", fontsize = 25)
ax.set_ylabel("Number of runs", fontsize = 25)
# ax.set_xticks(fontsize=16)
# ax.set_yticks(fontsize=16)
ax.set_title("Simulated activity of source", fontsize = 25)

textstr = '\n'.join((
    r'$Mean~activity = %.2f$' % (mean_activity, ),))

ax.hist(total_arr, bins=30)
# these are matplotlib.patch.Patch properties
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# # place a text box in upper left in axes coords
# ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
#         verticalalignment='top', bbox=props)
plt.tick_params(axis='both', which='major', labelsize=25)
ax.axvline(mean_activity, color='r', label=textstr, linestyle ='dashed')
ax.legend(fontsize=22)
plt.show()




# import numpy as np
# import matplotlib.pyplot as plt
 
# LENGTH_CYLINDER = 162 * 10**(-3)
# VECTOR_LENGTH = LENGTH_CYLINDER + 1



# def isotropic_vector_generator(r, n):
#     ''' Generates isotropic unit vectors given an array
#     Returns: Unit Vector'''
    
#     # First create random data in spherical polar coordinates
#     theta = np.arccos(np.random.uniform(-1, 1, n))  # Create random theta using arccos to avoid poles forming
#     phi = np.random.uniform(0, 2 * np.pi, n)

#     # Transform data into Cartesian coordinates
    
#     x = r * np.cos(phi) * np.sin(theta)
#     y = r * np.sin(phi) * np.sin(theta)
#     z = r * np.cos(theta)
    
#     return (x, y, z)

# fig = plt.figure()
# ax = plt.axes(projection ='3d')

# x_arr = []
# y_arr = []
# z_arr = []

# for i in range(1000):
#     x, y, z = isotropic_vector_generator(VECTOR_LENGTH,1)
#     x_arr.append(x)
#     y_arr.append(y)
#     z_arr.append(z)

# x_arr = np.array(x_arr)
# y_arr = np.array(y_arr)
# z_arr = np.array(z_arr)


# ax.scatter(x_arr, y_arr, z_arr)

# plt.show()


