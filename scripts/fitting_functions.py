import numpy as np
import scipy
import matplotlib.pyplot as plt
from datetime import datetime
import os
import pickle

def infer_params(means, covs, c_range, W_prior, fit_condition, N_datapoints,
                 N_fits=1000, prior_lower_bound=1e-3, fit_condition_tol=1e-5, 
                 solver_stopping_tol=1e-8, solver_max_iter=100000, random_prior=False, 
                 sample_correlated=False, verbose=True):

    params = []  # (W, h1, h0)
    errors = []
    
    for i in range(N_fits):
        if verbose:
            print(f'inferring param set {i+1}/{N_fits}')
        
        rates_sample = get_surrogate_data(means, covs, N_datapoints, sample_correlated=sample_correlated)
        
        W, h1, h0, final_error = get_params_nnls(
            rates_sample, c_range, W_prior, fit_condition, 
            prior_lower_bound=prior_lower_bound,
            fit_condition_tol=fit_condition_tol, 
            solver_stopping_tol=solver_stopping_tol,
            solver_max_iter=solver_max_iter, 
            random_prior=random_prior
        )
        
        params.append((W, h1, h0))
        errors.append(final_error)
    
    if verbose:
        print('parameter inference complete')
    
    return params, errors

def fit_rates(params, c_range, means, covs, fit_condition, 
                    N_fits=1000, z_lims=(-10, 50), N_z=100000, sample_correlated=False, verbose=True):
    if N_fits is None:
        N_fits = len(params)
    
    fits = []  # list of fits (4,6)
    likelihoods = []  # log likelihoods under original data
    z_roots_lists = []  # list (N_f) of lists (6) of list of roots
    
    for i in range(N_fits):
        if verbose:
            print(f'fitting rates {i+1}/{N_fits}')
        
        W, h1, h0 = params[i]
        fit, *_, z_roots_list = get_full_fit_poly(
            W, h1, h0, c_range, means, fit_condition,
            z_lims=z_lims, N_z=N_z, 
        )
        
        log_lik = get_log_likelihood(fit, means, covs, sample_correlated=sample_correlated)
        
        fits.append(fit)
        likelihoods.append(log_lik)
        z_roots_lists.append(z_roots_list)
    
    fits = np.array(fits)
    likelihoods = np.array(likelihoods)
    
    if verbose:
        no_nan_arrays = [arr for arr in fits if not np.isnan(arr).any()]
        fraction_no_nans = len(no_nan_arrays) / len(fits)
        print(f'fitting complete\nfraction of param sets with valid fits: {fraction_no_nans:.4f}')
    
    return fits, likelihoods, z_roots_lists

def filter_fits(params, fits, likelihoods, c_range, TAU, 
                filter_nan=True, filter_ISN=True, filter_unstable=True, verbose=True):
    filtered_params = []
    filtered_fits = []
    filtered_likelihoods = []
    
    N_fits = len(fits)
    nan_count = 0
    ISN_count = 0
    unstable_count = 0
    
    for i in range(N_fits):
        if verbose:
            print(f'filtering fit {i+1}/{N_fits}')
            
        (W, h1, h0), fit, log_lik = params[i], fits[i], likelihoods[i]
        
        if filter_nan and np.isnan(fit).any(): # filter out nans
            if verbose:
                print(f'fit {i} filtered out: NaNs in fit')
            nan_count += 1
            continue
            
        if filter_ISN: # filter out negative ISN coefficients
            ISN_coeffs = get_ISN_coeffs(W, h1, h0, c_range, fit)
            if np.any(ISN_coeffs < 1):
                if verbose:
                    print(f'fit {i} filtered out: negative ISN coeff')
                ISN_count += 1
                continue
                
        if filter_unstable: # filter out unstable steady states
            stable_fit = True
            for j in range(len(c_range)):
                fit_c = fit[:,j]
                contrast = c_range[j]
                f_grad, X, J = linearise(fit_c, W, h1, h0, contrast, TAU)
                eigenvalues = np.linalg.eigvals(J)
                if np.any(eigenvalues > 0):
                    if verbose:
                        print(f'fit {i} filtered out: positive eigenvalues')
                    stable_fit = False
                    break
                    
            if not stable_fit:
                unstable_count += 1
                continue
                
        filtered_params.append((W, h1, h0))
        filtered_fits.append(fit)
        filtered_likelihoods.append(log_lik)
    
    if verbose:
        print(f'total fits after filtering: {len(filtered_fits)}/{N_fits}')
        if filter_nan:
            print(f'  removed due to NaNs: {nan_count}')
        if filter_ISN:
            print(f'  removed due to negative ISN coefficients: {ISN_count}')
        if filter_unstable:
            print(f'  removed due to instability: {unstable_count}')
        
    return filtered_params, filtered_fits, filtered_likelihoods

def get_surrogate_data(means, covariances, N_datapoints, sample_correlated=False):
    rates_sample = []

    for alpha in range(means.shape[0]): 
        mean = means[alpha]
        cov = covariances[alpha]
        T = N_datapoints[alpha] 
        
        if sample_correlated:
            scaled_cov = cov / T
            sampled = np.random.multivariate_normal(mean, scaled_cov)
        else:
            variances = np.diag(cov)
            sems = np.sqrt(variances / T)
            sampled = np.random.normal(mean, sems)
        
        rates_sample.append(sampled)

    return np.array(rates_sample)

def get_params_nnls(r_sample, c_range, W_prior, fit_condition, prior_lower_bound=1e-3, fit_condition_tol=1e-5, solver_stopping_tol=1e-8, solver_max_iter=100000, random_prior=False):
    def assemble_design_matrix(r_sample, c_range, N_c, N_a):
        X = np.zeros((N_c*N_a, N_a**2 + 2*N_a))
        for i in range(N_c):
            r = r_sample[:,i]
            c = c_range[i]
            X[i*N_a:(i+1)*N_a, :N_a**2] = np.kron(np.eye(N_a), r) 
            X[i*N_a:(i+1)*N_a, N_a**2:N_a**2+N_a] = c * np.eye(N_a) 
            X[i*N_a:(i+1)*N_a, N_a**2+N_a:] = np.eye(N_a) 
        return X
    
    def objective_function(beta, X, Y):
        return np.sum((Y - X @ beta) ** 2)
    
    def product_constraint_ineq(params):
        W = params[:N_a**2].reshape(N_a, N_a)
        if fit_condition == 1:
            prod = W[0,2] * W[1,1] - W[0,1] * W[1,2]
        elif fit_condition == 2:
            prod = W[0,2] * W[1,0] - W[0,0] * W[1,2]
        else:
            raise ValueError("unsupported fit_condition")
        
        return [
            prod + fit_condition_tol,  # product >= -tolerance
            -prod + fit_condition_tol  # product <= tolerance
        ]
    
    N_a, N_c = r_sample.shape
    y_samples = r_sample ** (1/2)
    
    X = assemble_design_matrix(r_sample, c_range, N_c, N_a)
    Y = y_samples.T.flatten()
    flat_prior = W_prior.flatten()
    
    N_h = 2 * N_a
    
    bounds = []
    for prior_val in flat_prior:
        if prior_val > 0:
            # [lower_bound, prior_val]
            bounds.append((prior_lower_bound, prior_val))
        elif prior_val < 0:
            # [prior_val, -lower_bound]
            bounds.append((prior_val, -prior_lower_bound))
        else:
            bounds.append((0, 0))
    
    for i in range(N_h):
        if i == 2 or i == 3:
            bounds.append((0, 0))  # no contrast input to S/V
        else:
            bounds.append((0, None))  # positive bounds for h0/h1
    
    W0 = np.zeros(N_a**2)
    for i, prior_val in enumerate(flat_prior):
        if prior_val > 0:
            if random_prior:
                W0[i] = np.random.uniform(prior_lower_bound, prior_val)  # random val within bounds
            else:
                W0[i] = prior_val / 2  # middle of allowed range
        elif prior_val < 0:
            if random_prior:
                W0[i] = np.random.uniform(prior_val, -prior_lower_bound)  # random val within bounds
            else:
                W0[i] = prior_val / 2  # middle of allowed range
    
    h0 = np.zeros(N_h)
    h0[2:4] = 0  # no contrast input to S/V
    if random_prior:
        h0[:2] = np.random.uniform(prior_lower_bound, np.abs(max(W_prior)), size=2)
        h0[4:] = np.random.uniform(prior_lower_bound, np.abs(max(W_prior)), size=N_a) 
    else:
        h0[:2] = W_prior[0,0] / 2 
        h0[4:] = W_prior[0,0] / 2 
    x0 = np.concatenate([W0, h0])
    
    constraint_ineq = [
        {'type': 'ineq', 'fun': lambda params: product_constraint_ineq(params)[0]},  # lower bound
        {'type': 'ineq', 'fun': lambda params: product_constraint_ineq(params)[1]}   # upper bound
    ]
    
    result = scipy.optimize.minimize(
        objective_function,
        x0,
        args=(X, Y),
        method='SLSQP',
        bounds=bounds,
        constraints=constraint_ineq,
        tol=solver_stopping_tol,  # stopping tolerance
        options={
            'maxiter': solver_max_iter,
        }
    )
    
    final_constraint_vals = product_constraint_ineq(result.x)
    violations = [val for val in final_constraint_vals if val < 0]  # violations are negative
    if violations:
        print(f"constraint violation by {violations}")
    else:
        print("all constraints within tolerance")
    
    params = result.x
    W_opt = params[:N_a**2].reshape(N_a, N_a)
    h1_opt = params[N_a**2:N_a**2+N_a]
    h0_opt = params[N_a**2+N_a:]
    final_error = result.fun
    return W_opt, h1_opt, h0_opt, final_error

def find_zero_crossings(x, y):
    signs = np.sign(y)
    zero_crossings = []
    zc_derivs = []
    
    for i in range(len(y)-1):
        if signs[i] * signs[i+1] <= 0:
            x_cross = x[i] - y[i] * (x[i+1] - x[i])/(y[i+1] - y[i])
            zero_crossings.append(x_cross)
            if y[i] > 0:
                zc_derivs.append(0) # -ve grad crossing (stable)
            else:
                zc_derivs.append(1) # +ve grad crossing (unstable)

    return zero_crossings, zc_derivs

def error_fit(W, h1, h0, contrast, fp_c, fit_c, f):
    z = W @ fp_c + h1 * contrast + h0
    r_new = f(z)
    return np.sum((r_new - fit_c)**2) # sum

def get_fit_poly_1(W, h1, h0, contrast, c_range, means, z_lims=[-10,50], N_z=100000):

    W_EE, W_EP, W_ES, W_EV = W[0, 0], -W[0, 1], -W[0, 2], -W[0, 3]
    W_PE, W_PP, W_PS, W_PV = W[1, 0], -W[1, 1], -W[1, 2], -W[1, 3]
    W_SE, W_SP, W_SS, W_SV = W[2, 0], -W[2, 1], -W[2, 2], -W[2, 3]
    W_VE, W_VP, W_VS, W_VV = W[3, 0], -W[3, 1], -W[3, 2], -W[3, 3]

    cond_a = W_ES * W_PP 
    cond_b = W_EP * W_PS
    c_ind = np.where(contrast == c_range)[0][0]

    h_E, h_P, h_S, h_V = h1 * contrast + h0

    def f(x, exp=2):
        return np.maximum(x,0) ** exp

    def z_V_of_z_E(z_E):
        return (
            (W_VE * W_ES - W_VS * W_EE) / W_ES * f(z_E) 
            + (W_VS * W_EP - W_VP * W_ES) / W_ES * f(z_P_of_z_E(z_E)) 
            + W_VS / W_ES * (z_E - h_E) 
            + h_V
        )

    def z_S_of_z_E(z_E):
        return (
            (W_ES * W_SE - W_EE * W_SS) / W_ES * f(z_E) 
            + (W_EP * W_SS - W_ES * W_SP) / W_ES * f(z_P_of_z_E(z_E)) - W_SV * f(z_V_of_z_E(z_E)) 
            + h_S + W_SS / W_ES * (z_E - h_E)
        )
    
    def z_P_of_z_E(z_E):
        return (
            (-W_PP * W_EE + W_EP * W_PE) / W_EP * f(z_E) 
            + W_PP / W_EP * (z_E - h_E) 
            + h_P
        )

    z_E = np.linspace(z_lims[0], z_lims[1], N_z)
    F_z = W_EE * f(z_E) - W_EP * f(z_P_of_z_E(z_E)) - W_ES * f(z_S_of_z_E(z_E)) - z_E + h_E

    z_roots, z_derivs = find_zero_crossings(z_E, F_z)

    fixed_points = []
    fp_errors = []
    fp_labels = []
    z_root_list = []
    for z_root, z_deriv in zip(z_roots, z_derivs):
        z_root_list.append(z_root)

        z_P_root = z_P_of_z_E(z_root)
        z_S_root = z_S_of_z_E(z_root)
        z_V_root = z_V_of_z_E(z_root)
        fp = np.array([
            f(z_root),
            f(z_P_root),
            f(z_S_root),
            f(z_V_root)
        ])

        fixed_points.append(fp)
        error = error_fit(W, h1, h0, contrast, fp, means[:,c_ind], f)
        fp_errors.append(error)
        if z_deriv == 0:
            fp_labels.append(0)
        else:
            fp_labels.append(1)
    if len(fixed_points) == 0:
        fixed_points.append([np.nan, np.nan, np.nan, np.nan])
        main_fp = [np.nan, np.nan, np.nan, np.nan]
    indices_with_label_0 = [i for i, label in enumerate(fp_labels) if label == 0] or None
    if indices_with_label_0 is None:
        print('no stable fixed points')
    else:
        errors_with_label_0 = [fp_errors[i] for i in indices_with_label_0]
        min_ind = indices_with_label_0[np.argmin(errors_with_label_0)]
        main_fp = fixed_points[min_ind]
        # main_fp = fixed_points[indices_with_label_0[0]] # take main fp as first stable fp (lowest z crossing)
    
    return main_fp, fixed_points, fp_errors, fp_labels, z_E, F_z, z_root_list, cond_a, cond_b

def get_fit_poly_2(W, h1, h0, contrast, c_range, means, z_lims=[-10,50], N_z=100000):

    W_EE, W_EP, W_ES, W_EV = W[0, 0], -W[0, 1], -W[0, 2], -W[0, 3]
    W_PE, W_PP, W_PS, W_PV = W[1, 0], -W[1, 1], -W[1, 2], -W[1, 3]
    W_SE, W_SP, W_SS, W_SV = W[2, 0], -W[2, 1], -W[2, 2], -W[2, 3]
    W_VE, W_VP, W_VS, W_VV = W[3, 0], -W[3, 1], -W[3, 2], -W[3, 3]

    cond_a = W_ES * W_PP
    cond_b = W_EP * W_PS
    c_ind = np.where(contrast == c_range)[0][0]

    h_E, h_P, h_S, h_V = h1 * contrast + h0

    def f(x, exp=2):
        return np.maximum(x,0) ** exp

    def z_E_of_z_P(z_P):
        return (
            ((W_EE * W_PP - W_EP * W_PE) / W_PE) * f(z_P)
            + (W_EE / W_PE) * (z_P - h_P)
            + h_E
        )

    def z_V_of_z_P(z_P):
        return (
            ((W_VE * W_PS - W_VS * W_PE) / W_PS) * f(z_E_of_z_P(z_P))
            + ((W_VS * W_PP - W_VP * W_PS) / W_PS) * f(z_P)
            + (W_VS / W_PS) * (z_P - h_P)
            + h_V
        )

    def z_S_of_z_P(z_P):
        return (
            ((W_PS * W_SP - W_PP * W_SS) / W_PS) * f(z_P)
            + ((W_PS * W_SE - W_PS * W_SS) / W_PS) * f(z_E_of_z_P(z_P))
            - (W_PS * W_SV / W_PS) * f(z_V_of_z_P(z_P))
            + h_S
            + W_SS / W_PS * (z_P - h_P)
        )

    z_P = np.linspace(z_lims[0], z_lims[1], N_z)
    F_z = W_PE * f(z_E_of_z_P(z_P)) - W_PP * f(z_P) - W_PS * f(z_S_of_z_P(z_P)) - z_P + h_P

    z_roots, z_derivs = find_zero_crossings(z_P, F_z)
    fixed_points = []
    fp_labels = []
    fp_errors = []
    z_root_list = []
    for z_root, z_deriv in zip(z_roots, z_derivs):
        z_root_list.append(z_root)

        z_E_root = z_E_of_z_P(z_root)
        z_S_root = z_S_of_z_P(z_root)
        z_V_root = z_V_of_z_P(z_root)
        
        fp = np.array([
            f(z_E_root),
            f(z_root),
            f(z_S_root),
            f(z_V_root)
        ])

        fixed_points.append(fp)
        error = error_fit(W, h1, h0, contrast, fp, means[:,c_ind], f)
        fp_errors.append(error)
        if z_deriv == 0:
            fp_labels.append(0)
        else:
            fp_labels.append(1)
    if len(fixed_points) == 0:
        fixed_points.append([np.nan, np.nan, np.nan, np.nan])
        main_fp = [np.nan, np.nan, np.nan, np.nan]
    indices_with_label_0 = [i for i, label in enumerate(fp_labels) if label == 0] or None
    if indices_with_label_0 is None:
        print('no stable fixed points')
    else:
        errors_with_label_0 = [fp_errors[i] for i in indices_with_label_0]
        min_ind = indices_with_label_0[np.argmin(errors_with_label_0)]
        main_fp = fixed_points[min_ind] # main fp closest to previous (fit_c_1)
        # main_fp = fixed_points[indices_with_label_0[0]] # take main fp as first stable fp (lowest z crossing)
    
    return main_fp, fixed_points, fp_errors, fp_labels, z_P, F_z, z_root_list, cond_a, cond_b

def get_ss_linearise(fit_c_1, W, h1, h0_tot, contrast, TAU, z_lims=[-10,50], N_z=100000):
    main_fp, main_fp_error = get_ss(fit_c_1, W, h1, h0_tot, contrast, z_lims, N_z)
    main_fp_refined = fixed_point_refinement(main_fp, W, h1, h0_tot, contrast)
    f_grad, X, J = linearise(main_fp_refined, W, h1, h0_tot, contrast, TAU)
    return f_grad, X, J, main_fp_refined, main_fp_error

def get_ss(fit_c_1, W, h1, h0, contrast, z_lims=[-10,50], N_z=100000):
    """
    stripped down version of get_fit_poly_1 which finds fp closest to previous fit
    """
    W_EE, W_EP, W_ES, W_EV = W[0, 0], -W[0, 1], -W[0, 2], -W[0, 3]
    W_PE, W_PP, W_PS, W_PV = W[1, 0], -W[1, 1], -W[1, 2], -W[1, 3]
    W_SE, W_SP, W_SS, W_SV = W[2, 0], -W[2, 1], -W[2, 2], -W[2, 3]
    W_VE, W_VP, W_VS, W_VV = W[3, 0], -W[3, 1], -W[3, 2], -W[3, 3]

    h_E, h_P, h_S, h_V = h1 * contrast + h0
    
    def f(x, exp=2):
        return np.maximum(x,0) ** exp

    def z_V_of_z_E(z_E):
        return (
            (W_VE * W_ES - W_VS * W_EE) / W_ES * f(z_E) 
            + (W_VS * W_EP - W_VP * W_ES) / W_ES * f(z_P_of_z_E(z_E)) 
            + W_VS / W_ES * (z_E - h_E) 
            + h_V
        )

    def z_S_of_z_E(z_E):
        return (
            (W_ES * W_SE - W_EE * W_SS) / W_ES * f(z_E) 
            + (W_EP * W_SS - W_ES * W_SP) / W_ES * f(z_P_of_z_E(z_E)) - W_SV * f(z_V_of_z_E(z_E)) 
            + h_S + W_SS / W_ES * (z_E - h_E)
        )
    
    def z_P_of_z_E(z_E):
        return (
            (-W_PP * W_EE + W_EP * W_PE) / W_EP * f(z_E) 
            + W_PP / W_EP * (z_E - h_E) 
            + h_P
        )

    z_E = np.linspace(z_lims[0], z_lims[1], N_z)
    F_z = W_EE * f(z_E) - W_EP * f(z_P_of_z_E(z_E)) - W_ES * f(z_S_of_z_E(z_E)) - z_E + h_E

    z_roots, z_derivs = find_zero_crossings(z_E, F_z)

    fixed_points = []
    fp_errors = []
    z_root_list = []
    for z_root, z_deriv in zip(z_roots, z_derivs):
        z_root_list.append(z_root)

        z_P_root = z_P_of_z_E(z_root)
        z_S_root = z_S_of_z_E(z_root)
        z_V_root = z_V_of_z_E(z_root)
        fp = [
            f(z_root),
            f(z_P_root),
            f(z_S_root),
            f(z_V_root)
        ]

        error = error_fit(W, h1, h0, contrast, fp, fit_c_1, f)
        fixed_points.append(fp)
        fp_errors.append(error)

    if len(fixed_points) == 0:
        main_fp = [np.nan, np.nan, np.nan, np.nan]
        main_fp_error = np.nan
    else:
        main_fp = fixed_points[np.argmin(fp_errors)] # main fp closest to previous (fit_c_1)
        main_fp_error = np.min(fp_errors)

    main_fp_refined = fixed_point_refinement(main_fp, W, h1, h0, contrast)
    
    return main_fp_refined, main_fp_error

def fixed_point_residual(r, W, h1, h0, c, r_init, lambda_penalty=1e-4):
    penalty_weight = lambda_penalty * np.linalg.norm(r_init)
    x = W @ r + h1 * c + h0
    f_r = np.where(x > 0, x**2, 0)
    residual = np.sum((f_r - r)**2)
    penalty = penalty_weight * np.sum((r - r_init)**2)  # penalise deviation from r_init
    return residual + penalty

def fixed_point_refinement(r_init, W, h1, h0, c, delta_mag=1e-1, maxiter=50000, ftol=1e-8):
    delta = delta_mag * np.max(r_init)
    bounds = [(max(r - delta, 0), r + delta) for r in r_init]
    result = scipy.optimize.minimize(
        fixed_point_residual, 
        r_init, 
        args=(W, h1, h0, c, r_init),
        method='L-BFGS-B', 
        bounds=bounds,
        options={'maxiter': maxiter, 'ftol': ftol}
    )
    if result.success:
        # print(f"fp refinement converged in {result.nit} its")
        pass
    else:
        print(f"fp refinement failed: {result.message}")

    return result.x

def get_full_fit_poly(W, h1, h0, c_range, means, fit_condition, z_lims=[-10,50], N_z=100000):
    r_fit = []
    r_fits_poly = []
    fp_errors_list = []
    fp_labels_list = []
    cond_list = []
    z_roots_list = []
    for i, contrast in enumerate(c_range): 
        if fit_condition == 1:
            main_fp, fixed_points, fp_errors, fp_labels, z, F_z, z_roots, cond_a, cond_b = get_fit_poly_1(W, h1, h0, contrast, c_range, means, z_lims=z_lims, N_z=N_z)
        elif fit_condition == 2:
            main_fp, fixed_points, fp_errors, fp_labels, z, F_z, z_roots, cond_a, cond_b = get_fit_poly_2(W, h1, h0, contrast, c_range, means, z_lims=z_lims, N_z=N_z)
        main_fp_refined = fixed_point_refinement(main_fp, W, h1, h0, contrast)
        r_fits_poly.append(fixed_points)
        fp_errors_list.append(fp_errors)
        fp_labels_list.append(fp_labels)
        cond_list.append([cond_a, cond_b])
        r_fit.append(main_fp_refined)
        z_roots_list.append(z_roots)
    r_fit = np.array(r_fit).T

    return r_fit, r_fits_poly, cond_list, fp_errors_list, fp_labels_list, z_roots_list 

def get_log_likelihood(fit, means, covs, sample_correlated=True):
    N_a, N_c = means.shape 
    log_likelihood = 0

    for alpha in range(N_a):
        diff = fit[alpha] - means[alpha]
        cov = covs[alpha]
        
        if sample_correlated:
            inv_cov = np.linalg.inv(cov)
            log_det_cov = np.log(np.linalg.det(cov))
            normalization = -0.5 * (N_c * np.log(2 * np.pi) + log_det_cov)
            quadratic_form = -0.5 * diff.T @ inv_cov @ diff
            log_likelihood += normalization + quadratic_form
        else:
            variances = np.diag(cov) 
            normalization = -0.5 * np.sum(np.log(2 * np.pi * variances))
            quadratic_form = -0.5 * np.sum((diff**2) / variances)
            log_likelihood += normalization + quadratic_form
    
    return log_likelihood

def get_ISN_coeffs(W, h1, h0, c_range, rates):
    N_a, N_c = rates.shape
    z = (W @ rates + np.outer(h1, c_range) + np.tile(h0.reshape(N_a, 1), (1, N_c)))
    f_grad = 2 * np.maximum(z, 0)
    ISN_coeffs = f_grad[0,:] * W[0,0] # f'_E * W_EE
    return ISN_coeffs

def linearise(fit_c, W, h1, h0, contrast, TAU):
    d = W.shape[0]
    h_tot = h1 * contrast + h0
    z = W @ fit_c + h_tot
    f_grad = 2 * np.maximum(z, 0) # 2 * z
    f_grad_safe = np.where(f_grad <= 0, 1e-8, f_grad) # 1e-10
    X = np.linalg.inv(np.linalg.inv(np.diag(f_grad_safe)) - W)
    J = np.linalg.inv(TAU) @ (-np.eye(d) + np.diag(f_grad_safe) @ W)
    return f_grad_safe, X, J

def get_top_k_fits(filtered_fits, filtered_params, filtered_likelihoods, top_k, fit_condition, save_to_file=True):
    top_indices = np.argsort(filtered_likelihoods)[-top_k:][::-1]  # sort descending
    actual_k = min(top_k, len(filtered_fits))

    top_fits = [filtered_fits[i] for i in top_indices]
    top_params = [filtered_params[i] for i in top_indices]
    top_likelihoods = [filtered_likelihoods[i] for i in top_indices]

    print(f'found top {actual_k} fits/params')

    if save_to_file:
        datet = datetime.now().strftime("%y%m%d_%H%M%S")
        file_path = f'../data/fits_cnd{fit_condition}_top{actual_k}_{datet}.pkl'
        with open(file_path, 'wb') as f:
            pickle.dump((top_fits, top_params, top_likelihoods), f)
        print(f"saved top {actual_k} fits at {file_path}")
    
    return top_fits, top_params, top_likelihoods

def get_scond_0(J):
    return -np.linalg.trace(J)

def get_scond_1(J):
    return np.linalg.trace(J@J@J) - np.linalg.trace(J)**3

def get_scond_2(TAU, f_grad, X, J):
    temp_A = np.sum( np.divide (np.diag(TAU), f_grad) * np.diag(X))
    temp_B = np.divide( 3 * np.linalg.trace(J)**2, np.linalg.trace(J@J@J) - np.linalg.trace(J)**3)
    return temp_A - temp_B

def get_scond_3(J):
    return np.linalg.det(J)

def get_fp_condition(fp, W, h1, h0, contrast, TAU):
    f_grad, X, J = linearise(fp, W, h1, h0, contrast, TAU)
    c0 = get_scond_0(J)
    c1 = get_scond_1(J)
    c2 = get_scond_2(TAU, f_grad, X, J)
    c3 = get_scond_3(J)
    routh_hurwitz = [c0, c1, c2, c3]
    if np.all(np.array(routh_hurwitz) > 0):
        condition = 0 # stable
    elif np.all(np.array(routh_hurwitz) < 0):
        condition = 1 # unstable
    else:
        condition = 2 # saddle
    return condition, J

