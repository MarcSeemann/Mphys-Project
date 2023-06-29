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


time_values = np.logspace(-2, 3, 10000)


# y_val = V_L_model(time_values, 0.21, 5.6*10**(-3), 0.79, 0, 0.08, 200, 10**(-5))

# guess = [0.21, 5.6*10**(-3), 0.79, 0, 0.08, 200, 10**(-5)]


# y_val = V_L_model(time_values, 0.89, 8.3*10**(-3), 0.11, 0.9, 0.2,1600, 10**(-6))

# guess = [0.89, 8.3*10**(-3), 0.11, 0.9, 0.2,1600, 10**(-6)]



# y_val = V_L_model(time_values, 0.72, 4.2*10**(-3), 0.28, 0.02, 0.064,1000, 10**(-5))

# guess = [0.72, 4.2*10**(-3), 0.28, 0.02, 0.064,1000, 10**(-5)]




# y_val = V_L_model(time_values, 0.74, 4.6*10**(-3), 0.26, 0.19, 0.095,350, 10**(-5))

# guess = [0.74, 4.6*10**(-3), 0.26, 0.19, 0.095,350, 10**(-5)]

# ax.plot(time_values, y_val, label='77k')
# ax.plot(time_values, y_val, label='295k')
# ax.set_ylim([10**(-7), 100])
# ax.set_xlim([10**(-2), 1000])
# ax.set_yscale("log")
# ax.set_xscale("log")
# ax.grid()
# plt.legend()
# plt.show()

# def func_1(t):
#     t_a = 0.2
#     tau_t = 1600
#     return -1*expi((1 * ((t + t_a)/tau_t)))

def func_2(t):
    t_a = 10
    tau_t = 1600
    A = 0.9
    return np.exp((-2*t)/tau_t) / (((1 + A*(1*expi((1 * ((t + t_a)/tau_t))) - (1*expi(t_a/tau_t))))**2) * (1+(t/t_a)))

# plt.plot(time_values, func_2(time_values))
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(10**(-2), 10**(3))
# plt.show()



# def V_L_fitting_func(t, N_p, tau_s, integrated_intens_s, N_d, tau_d, tau_t, C):
#     R_func = 1
#     return R_func * (prompt_func(t, N_p, tau_s) + slow_func(t, N_d, tau_d, tau_t)) + C


def fit_func(t, N_p, tau_s, N_d, A, t_a, tau_t, C):
    return V_L_model(t, N_p, tau_s, N_d, A, t_a, tau_t, C)




# def fit_data(x, y, p0):
#     popt, pcov = curve_fit(fit_func, x, y, p0)
#     return popt, pcov
# popt, pcov = fit_data(time_values, y_val, guess)




# fig, ax = plt.subplots()
# ax.plot(time_values, y_val)
# ax.set_ylim([10**(-7), 100])
# ax.set_xlim([10**(-2), 1000])
# ax.set_yscale("log")
# ax.set_xscale("log")
# ax.grid()
# plt.show()



# def mc_gen(time):
#     data = []
#     for i in time:
#         data.append(V_L_model(i, 0.7, 5.0, 0.8, 500, 0.01, 0.1) + random.uniform(-1, 1))
#     return np.array(data)

# mc_data = mc_gen(time_values)
# plt.plot(time_values, mc_data)
# plt.plot(time_values, y_val)
# plt.show()


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

ax.set_xlabel(r"Time [$\mu s$]", fontsize = 25)
ax.set_ylabel(r"Amplitude", fontsize = 25)
ax.tick_params(axis='both', which='major', labelsize=20)
# ax.hist(sampled_part1, bins=bin_widths1)
# ax.hist(sampled_part2, bins=bin_widths2)
# ax.errorbar(new_x_1, new_y_1, yerr=np.sqrt(new_y_1), label='first')
# ax.errorbar(new_x_2, new_y_2/535, yerr=np.sqrt(new_y_2)/535, label='second')

# ax.errorbar(comb_x, comb_y, yerr=comb_e, label=r'Monte Carlo data', alpha=0.5)
# ax.plot(time_values, data_y*2200, label=r'V&L model')

ax.errorbar(comb_x, comb_y/2200, yerr=comb_e/2200, label=r'Monte Carlo data', alpha=0.5)
ax.plot(time_values, data_y, label=r'V&L model')

ax.set_ylim([10**(-7), 100])
ax.set_xlim([10**(-2), 1000])
ax.set_yscale("log")
ax.set_xscale("log")
ax.grid()
plt.legend(fontsize='20')
plt.show()




# textstr = '\n'.join((
#     r'$A=%.2f$ $\pm~%.3f$' % (popt[0], perr[0], ),
#     r'$\mu=%.2f$ $\pm~%.3f$' % (popt[1], perr[1], ),
#     r'$\sigma=%.2f$ $\pm~%.3f$' % (popt[2], perr[2], )))

# these are matplotlib.patch.Patch properties
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# # place a text box in upper left in axes coords
# ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=20,
#         verticalalignment='top', bbox=props)
# ax.set_yscale("log")
# ax.set_xlim([0, 4000])
# ax.legend(fontsize='20')
# plt.show()
def V_L_model_new(t, N_p, tau_s, N_d, A, t_a, tau_t, C):
    return (prompt_func(t, N_p, tau_s) + slow_func(t, N_d, tau_t, t_a, A) + C)*2200


def fit_model(x, y, p0, yerr):
    popt, pcov = curve_fit(V_L_model_new, x, y, p0, sigma=yerr, maxfev=5000)
    return popt, pcov

guess = [0.72, 4.2*10**(-3), 0.28, 0.02, 0.064,1000, 10**(-5)]

popt, pcov = fit_model(comb_x, comb_y, guess, comb_e)
perr = np.sqrt(np.diag(pcov))


def chi_squared(x_data, y_data, param, func):
    yfit = func(x_data, *param)
    residuals = y_data - yfit
    chisq = np.sum((residuals)**2/yfit)
    dof = len(y_data) - len(param)
    redchisq = chisq / dof
    return chisq, redchisq

chi_sq_var = chi_squared(comb_x, comb_y, popt, V_L_model_new)





xspace = np.logspace(-2, 3, 1000)

fig, ax = plt.subplots()

ax.errorbar(comb_x, comb_y, yerr=comb_e, label=r'Monte Carlo data', alpha=0.5)
ax.plot(xspace, V_L_model_new(xspace, *popt), color='orange', linewidth=2.5, label=r'Fitted function',linestyle='dashed')

ax.set_xlabel(r"Time [$\mu s$]", fontsize = 25)
ax.set_ylabel(r"Amplitude", fontsize = 25)
ax.tick_params(axis='both', which='major', labelsize=20)


# textstr = '\n'.join((
#     r'$A=%.2f$ $\pm~%.3f$' % (popt[0], perr[0], ),
#     r'$\mu=%.2f$ $\pm~%.3f$' % (popt[1], perr[1], ),
#     r'$\sigma=%.2f$ $\pm~%.3f$' % (popt[2], perr[2], )))


textstr = '\n'.join((
    r'$\chi=%.2f$' % (chi_sq_var[0], ),
    r'$\chi_{red}=%.2f$' % (chi_sq_var[1], )))

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# place a text box in upper left in axes coords
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=20,
        verticalalignment='top', bbox=props)
ax.set_xlim([10**(-2), 1000])
ax.set_yscale("log")
ax.set_xscale("log")
ax.grid()
plt.legend(fontsize='20')
plt.show()



def diff_parameters_func(parameters, errors):
    diff = 0
    for i in range(len(guess)):
        diff += (guess[i] - parameters[i])/errors[i]
    return diff

print('Before_longcalc')

difference_arr = []
for i in range (200):
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
    
    def fit_model(x, y, p0, yerr):
        popt, pcov = curve_fit(V_L_model_new, x, y, p0, sigma=yerr, maxfev=5000)
        return popt, pcov
    guess = [0.72, 4.2*10**(-3), 0.28, 0.02, 0.064,1000, 10**(-5)]

    popt, pcov = fit_model(comb_x, comb_y, guess, comb_e)
    perr = np.sqrt(np.diag(pcov))
    
    # difference_arr.append(diff_parameters_func(popt, perr))
    difference_arr.append((guess[-1] - popt[-1])/perr[-1])
    print(i)




fig, ax = plt.subplots()

ax.hist(difference_arr, bins=50)

ax.set_xlabel(r"Difference in $\tau_{T}$ [$\mu s$]", fontsize = 25)
ax.set_ylabel(r"Counts", fontsize = 25)
ax.tick_params(axis='both', which='major', labelsize=20)


# textstr = '\n'.join((
#     r'$A=%.2f$ $\pm~%.3f$' % (popt[0], perr[0], ),
#     r'$\mu=%.2f$ $\pm~%.3f$' % (popt[1], perr[1], ),
#     r'$\sigma=%.2f$ $\pm~%.3f$' % (popt[2], perr[2], )))



# these are matplotlib.patch.Patch properties


# place a text box in upper left in axes coords

plt.legend(fontsize='20')
plt.show()


    