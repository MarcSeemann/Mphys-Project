import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.special import expi
from scipy.optimize import curve_fit
import random
from scipy.stats import rv_continuous



    
def integrand(t, tau_t, t_a, A):
    return np.exp((-2*t)/tau_t) / (((1 + A*(1*expi((1 * ((t + t_a)/tau_t))) - (1*expi(t_a/tau_t))))**2) * (1+(t/t_a)))

def tau_d_func(tau_t, t_a, A):
    result, _ = quad(integrand, 0, np.inf, args=(tau_t, t_a, A))
    return result


# def integrand_ei_first(t, t_a, tau_t):
#     return np.exp(-((t + t_a)/tau_t))/((t + t_a)/tau_t)

# def ei_func(t, t_a, tau_t):
#     result, _ = quad(integrand_ei_first, t, np.inf, args=(t_a, tau_t))
#     return result



def prompt_func(t, N_p, tau_s):
    return (N_p/tau_s) * np.exp(-t/tau_s)



def slow_func(t, N_d, tau_t, t_a, A):
    F_t = np.exp((-2*t)/tau_t) / (((1 + A*(1*expi((1 * ((t + t_a)/tau_t))) - (1*expi(t_a/tau_t))))**2) * (1+(t/t_a)))
    tau_d = tau_d_func(tau_t, t_a, A)
    # tau_d = 10
    return (N_d/tau_d) * F_t


# def slow_func(t, N_d, tau_t, t_a, A):
#     F_t = np.exp((-2*t)/tau_t) / (((1 + A*(1*ei_func((1 * ((t + t_a)/tau_t))) - (1*ei_func(t_a/tau_t))))**2) * (1+(t/t_a)))
#     tau_d = tau_d_func(tau_t, t_a, A)
#     # tau_d = 10
#     return (N_d/tau_d) * F_t



def V_L_model(t, N_p, tau_s, N_d, A, t_a, tau_t, C):
    return prompt_func(t, N_p, tau_s) + slow_func(t, N_d, tau_t, t_a, A) + C


# from scipy.stats import rv_continuous

# class v_l_generator(rv_continuous):
    
#     # def __init__(self, N_p, tau_s, N_d, A, t_a, tau_t, C):
#     #     self.N_p = N_p
#     #     self.tau_s = tau_s
#     #     self.N_d = N_d
#     #     self.A = A
#     #     self.t_a = t_a
#     #     self.tau_t = tau_t
#     #     self.C = C
    
#     # def V_L_model(self, t, N_p, tau_s, N_d, A, t_a, tau_t, C):
#     #     return prompt_func(t, N_p, tau_s) + slow_func(t, N_d, tau_t, t_a, A) + C
#     def V_L_model(self, t, N_p, tau_s, N_d, A, t_a, tau_t, C):
#         N_p=0.72
#         tau_s = 4.2*10**(-3)
#         N_d = 0.28
#         A = 0.02
#         t_a = 0.064
#         tau_t=1000
#         C=10**(-5)
#         return prompt_func(t, N_p, tau_s) + slow_func(t, N_d, tau_t, t_a, A) + C


# test_func = v_l_generator(name='test_func')
# x_data = np.logspace(-2, 3, 100)
# y_data_1 = test_func.V_L_model(x_data, N_p=0.72, tau_s = 4.2*10**(-3),N_d = 0.28,A = 0.02,t_a = 0.064, tau_t=1000, C=10**(-5))

# test_func_2 = v_l_generator(name='test_func_2')
# x_data = np.logspace(-2, 3, 100)
# y_data_2 = test_func_2.V_L_model(x_data, N_p=0.72, tau_s = 4.2*10**(-3),N_d = 0.28,A = 0.02,t_a = 0.064, tau_t=1000, C=10**(-5))


# x_data_special = np.logspace(-2, 3, 10000)
# y_data_numpy = np.random.choice(V_L_model(x_data_special, 0.72, 4.2*10**(-3), 0.28, 0.02, 0.064,1000, 10**(-5)),size=len(x_data_special))

# x_data = np.logspace(-2, 3, 100)

# fig, ax = plt.subplots()
# ax.scatter(x_data, y_data_1, marker='.', s=1, label='One')
# ax.scatter(x_data, y_data_2, marker='.', s=1, label='Two')
# ax.scatter(x_data_special, y_data_numpy, marker='.', s=1, label='numpy')
# ax.scatter(x_data, V_L_model(x_data, 0.72, 4.2*10**(-3), 0.28, 0.02, 0.064,1000, 10**(-5)), marker='.', s=1, label='three')
# ax.plot(x_data, V_L_model(x_data, 0.72, 4.2*10**(-3), 0.28, 0.02, 0.064,1000, 10**(-5)), label='Func', c='green', alpha=0.5)
# ax.set_ylim([10**(-7), 100])
# ax.set_xlim([10**(-2), 1000])
# ax.set_yscale("log")
# ax.set_xscale("log")
# ax.grid()
# plt.legend()
# plt.show()



def V_L_model(t, N_p, tau_s, N_d, A, t_a, tau_t, C):
    return prompt_func(t, N_p, tau_s) + slow_func(t, N_d, tau_t, t_a, A) + C

time_values = np.logspace(-2, 3, 1000)
data_y = V_L_model(time_values, 0.72, 4.2*10**(-3), 0.28, 0.02, 0.064,1000, 10**(-5))


time_values_part1 = np.logspace(-2, 1, 1000)
data_y_part1 = V_L_model(time_values_part1, 0.72, 4.2*10**(-3), 0.28, 0.02, 0.064,1000, 10**(-5))

time_values_part2 = np.logspace(0.5, 3, 1000)
data_y_part2 = V_L_model(time_values_part2, 0.72, 4.2*10**(-3), 0.28, 0.02, 0.064,1000, 10**(-5))


sampled_part1 = np.random.choice(time_values_part1, size=100000, p=data_y_part1/sum(data_y_part1))
sampled_part2 = np.random.choice(time_values_part2, size=100000, p=data_y_part2/sum(data_y_part2))

bin_widths1 = np.logspace(-2,1, 51)
bin_widths2 = np.logspace(0.5,3, 51)

new_y_1, new_x_1 = np.histogram(sampled_part1, bins=bin_widths1)
#new_x_1 = np.power(np.log10(new_x_1[:-1])+np.diff(np.log10(new_x_1)), 10)
new_x_1 = new_x_1[:-1]+np.diff(new_x_1)/2

new_y_2, new_x_2 = np.histogram(sampled_part2, bins=bin_widths2)
new_x_2 = new_x_2[:-1]+np.diff(new_x_2)/2

comb_x = np.concatenate([new_x_1[:-8], new_x_2])
comb_y = np.concatenate([new_y_1[:-8], new_y_2/535])
comb_e = (((comb_y*100)**0.5/100)**2+1/100000)**0.5   #poisson error + monte carlo

fig, ax = plt.subplots()



# ax.hist(sampled_part1, bins=bin_widths1)
# ax.hist(sampled_part2, bins=bin_widths2)
# ax.errorbar(new_x_1, new_y_1, yerr=np.sqrt(new_y_1), label='first')
# ax.errorbar(new_x_2, new_y_2/535, yerr=np.sqrt(new_y_2)/535, label='second')
ax.errorbar(comb_x, comb_y, yerr=comb_e, label='comb', alpha=0.5)
ax.plot(time_values, data_y*2200, label='fit')
# ax.set_ylim([10**(-7), 100])
ax.set_xlim([10**(-2), 1000])
ax.set_yscale("log")
ax.set_xscale("log")
ax.grid()
plt.legend()
plt.show()