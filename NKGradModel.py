# this code takes in phi and retention data and fits it to the NK model using scipy package
from scipy.optimize import curve_fit
import numpy as np


# model function to find k_eff
def find_keff(tg, kw, s1, s2, phi_i, phi_f, td, tm):
    tg = np.array(tg, dtype=np.float64)

    try:
        beta = (phi_f - phi_i) / tg

        k_i = kw * (((1 + (s2 * phi_i)) ** 2) * np.exp((-1 * s1 * phi_i) / (1 + (s2 * phi_i))))

        first_component = td / tm

        num_of_num = phi_i + (((1 + (s2 * phi_i)) / s1) * np.log(
            beta * kw * (s1 * (tm - (td / k_i))) * np.exp((-1 * s1 * phi_i) / (1 + (s2 * phi_i))) + 1))

        den_of_num = 1 - (((s2 * (1 + (s2 * phi_i))) / s1) * np.log(
            beta * kw * s1 * (tm - (td / k_i)) * np.exp((-1 * s1 * phi_i) / (1 + (s2 * phi_i))) + 1))

        num = (num_of_num / den_of_num) - phi_i

        den = beta * tm

        second_component = num / den

        k_eff = first_component + second_component
        return k_eff

    except Exception as e:
        print(f"Error in model: {e}")
        return np.full_like(tg, np.nan)


# fitting function
def get_nkGrad_params(phi_i, phi_f, td, tm, tex, tg_values, retention_time_values):
    k_ex_values = ((np.array(retention_time_values) - tex) - (tm - tex)) / (tm - tex)

    # Wrapper function so curve_fit sees only tg and the 3 parameters
    def wrapped_model(tg, kw, s1, s2):
        return find_keff(tg, kw, s1, s2, phi_i, phi_f, td, tm)

    # Initial guess for kw, s1, s2
    initial_guess = [10, 10, 1]
    lower_bounds = [0, 0, 0]
    upper_bounds = [np.inf, np.inf, np.inf]

    try:
        popt, pcov = curve_fit(wrapped_model, tg_values, k_ex_values, p0=initial_guess,
                               bounds=(lower_bounds, upper_bounds), max_nfev=20000)
        kw, S1, S2 = popt
        perr = np.sqrt(np.diag(pcov))

        return kw, S1, S2, perr

    except Exception as e:
        return f"Error in curve fitting: {e}"


if __name__ == '__main__':
    phi_i = 0.05
    phi_f = 0.95
    td = 0.02
    tex = 0
    tm = 0.1108
    tg = np.array([1, 3, 5, 9])
    retention_times = np.array([0.8314, 1.7753, 2.5341, 3.7763])


    print(get_nkGrad_params(phi_i, phi_f, td, tm, tex, tg, retention_times))

