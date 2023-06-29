# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import math
# import random
# import time
# import multiprocessing

# # Keep all constants in SI units (meters)

# RADIUS_SOURCE = 5 * 10**(-3)
# RADIUS_CYLINDER = 9.9 * 10 **(-3)
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





# def alpha_propagation(_):
#     n_absorb = 0
#     n_trans = 0
#     radius_source_squared = RADIUS_SOURCE ** 2
#     radius_cylinder_squared = RADIUS_CYLINDER ** 2
    
    
#     # for i in range(number_of_iter):
#     while(n_trans < 500):
#         # Create initial position for each particle
#         x_coord_0 = 0
#         y_coord_0 = 0
#         z_coord_0 = 0

#         # Create isotropic vector for particle and save new position
#         x, y, z = isotropic_vector_generator(VECTOR_LENGTH, 1)
#         x_coord_1 = x_coord_0 + x
#         y_coord_1 = y_coord_0 + y
#         z_coord_1 = z_coord_0 + z

#         # Find line parameters at heigh z = TPB surface
#         x_surface = ((LENGTH_CYLINDER - z_coord_0) * x_coord_1 / z_coord_1) + x_coord_0
#         y_surface = ((LENGTH_CYLINDER - z_coord_0) * y_coord_1 / z_coord_1) + y_coord_0
        
#         if x_surface ** 2 + y_surface ** 2 <= radius_cylinder_squared:
#             n_trans += 1
#         else:
#             n_absorb += 1
        
#     return n_trans, n_absorb



# # array_transmitted = []
# # array_activity = []

# # for i in range(10):
# #     number_transmitted, number_absorbed = alpha_propagation()
# #     array_transmitted.append(number_transmitted)
# #     array_activity.append(number_absorbed + number_transmitted)



# # plt.hist(array_activity, bins=50)
# # plt.show()






# # if __name__ == '__main__':
# #     num_processes = multiprocessing.cpu_count()
# #     pool = multiprocessing.Pool(processes=num_processes)
# #     num_tests = 10
# #     results1 = []
# #     results2 = []

# #     # Define a helper function to collect results
# #     def collect_results(result):
# #         result1, result2 = result
# #         results1.append(result1)
# #         results2.append(result2)

# #     # Map the test function to a range of values
# #     pool.map_async(alpha_propagation, range(num_tests), callback=collect_results)

# #     # Wait for all processes to complete
# #     pool.close()
# #     pool.join()

# #     # Do something with the results
# #     print(results1)
# #     print(results2)


# if __name__ == '__main__':
#     num_processes = multiprocessing.cpu_count()
#     pool = multiprocessing.Pool(processes=num_processes)
#     num_tests = 50
#     results = pool.map(alpha_propagation, [()] * num_tests)
#     results1 = [r[0] for r in results]
#     results2 = [r[1] for r in results]
#     pool.close()
#     pool.join()

#     # Do something with the results
#     print(results1)
#     print(results2)
    
#     plt.hist([x+1 for x in results2], bins=50)
#     plt.show()

# import numpy as np

# RYDBERG = 10973731.57

# n_2_arr = np.arange(4, 10)
# N_1_START = 3

# def wavelength(n_1, n_2):
#     lambda_val = 1/(RYDBERG * ((1/n_1**2) - (1/n_2**2)))
#     return lambda_val/10**(-9)


# for i in n_2_arr:
#     print(wavelength(N_1_START, i))

i = 2
file = open("PBS_" + str(i) + ".sh", "w+")
contents = str("0\n") + str(",\n") + str("1\n") +str("3")
file.write(contents)
file.close()


