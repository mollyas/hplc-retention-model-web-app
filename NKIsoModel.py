# this code takes in phi and retention data and fits it to the NK model using scipy package
from scipy.optimize import curve_fit
import numpy as np

# model function to find k
def find_k(phi, kw, S1, S2):
    phi = np.asarray(phi)/100

    term_for_log = 1 + S2 * phi
    if np.any(term_for_log <= 0):
        # If any element in term_for_log is non-positive, raise an error
        # Or you can choose to filter/replace these values, but raising is safer for now.
        raise ValueError(
            f"1 + S2 * phi resulted in non-positive values for log calculation. S2: {S2}, problematic phi values: {phi[term_for_log <= 0]}")

    try:
        ln_k = np.log(kw) + 2 * np.log(term_for_log) - (S1 * phi) / term_for_log
        k = np.exp(ln_k)
        return k
    except Exception as e:
        return f"Error: {e}"


# fitting function
def get_nkIso_params(phi_values, retention_times, tex, tm):
    k_values = ((retention_times - tex) - (tm - tex)) / (tm - tex)

    # initial guesses for kw, S1, and S2
    initial_guess = [10, 10, 1]
    lower_bounds = [0, 0, -1]
    upper_bounds = [np.inf, np.inf, np.inf]

    try:
        # fit and get parameters
        popt, pcov = curve_fit(find_k, phi_values, k_values, p0=initial_guess,
                               bounds=(lower_bounds, upper_bounds), max_nfev=20000)
        kw, S1, S2 = popt
        perr = np.sqrt(np.diag(pcov))

        return kw, S1, S2, perr

    except Exception:
        raise ValueError("Couldn't perform fit. Check to make sure all boxes are filled.")


if __name__ == '__main__':
    phi_values = np.array([87, 68.7, 57, 35.8, 33.1])
    retention_data = np.array([0.0431, 0.0504, 0.064, 0.2889, 0.431])
    tex = 0.0237
    tm = 0.0361

    print(get_nkIso_params(phi_values, retention_data, tex, tm))

