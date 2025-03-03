import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import os
import re
from datetime import datetime

from .fitting_functions import get_ss_linearise

def get_params(r_fp, W, TAU, tau_noise, sigma_noise, init_perturb_mag, dt, variable, noise_type):
    d = W.shape[0]
    f_grad = 2*np.diag(np.sqrt(r_fp)) # grad same for voltage/rate
    S = np.linalg.inv(TAU)@(np.eye(d) - W@f_grad) # jac (voltage form)

    SIGMA_noise = sigma_noise*np.eye(d) # std
    v0 = init_perturb_mag*np.random.uniform(-1, 1, size=d)

    if variable == 'voltage' and noise_type == 'white':
        A_00 = S
        A_01 = np.zeros((d,d))
        A_10 = np.zeros((d,d))
        A_11 = np.zeros((d,d))
        A = np.block([[A_00, A_01], [A_10, A_11]])
        B = np.block([[SIGMA_noise, np.zeros((d,d))], [np.zeros((d,d)), np.zeros((d,d))]]) # no sqrt(2/tau_n) term for white noise; note error in manuscript
        eta0 = np.random.multivariate_normal(mean=np.zeros(d), cov=SIGMA_noise**2*dt)
        x0 = np.concatenate([v0, eta0])

    elif variable == 'voltage' and noise_type == 'OU':
        A_00 = S
        A_01 = -np.linalg.inv(TAU)*np.eye(d)
        A_10 = np.zeros((d,d))
        A_11 = (1/tau_noise)*np.eye(d)
        A = np.block([[A_00, A_01], [A_10, A_11]])
        B = np.block([[np.zeros((d,d)), np.zeros((d,d))], [np.zeros((d,d)), np.sqrt(2/tau_noise)*SIGMA_noise]])
        eta0 = np.random.multivariate_normal(mean=np.zeros(d), cov=(2/tau_noise)*SIGMA_noise**2*dt)
        x0 = np.concatenate([v0, eta0])

    elif variable == 'rate' and noise_type == 'OU':
        A_00 = S
        A_01 = -np.linalg.inv(tau_noise*TAU)@(tau_noise - TAU)
        A_10 = np.zeros((d,d))
        A_11 = (1/tau_noise)*np.eye(d)
        A = np.block([[A_00, A_01], [A_10, A_11]])
        B = np.block([[np.zeros((d,d)), np.sqrt(2/tau_noise)*SIGMA_noise], [np.zeros((d,d)), np.sqrt(2/tau_noise)*SIGMA_noise]])
        eta0 = np.random.multivariate_normal(mean=np.zeros(d), cov=(2/tau_noise)*SIGMA_noise**2*dt)
        x0 = np.concatenate([v0, eta0])

    else:
        raise ValueError('invalid variable/noise type combination')
    
    return A, B, x0

def sim_local(r_fp, init_perturb, W, h_tot, A, B, variable, noise_type, T_sim=10, dt=1e-4):
    d = A.shape[0]
    d_eta = d//2
    tot_steps = int(T_sim/dt)
    x_trajectory = np.zeros((tot_steps, d))
    x = init_perturb.copy()
    x_trajectory[0] = x
    blown_up = False
    
    if noise_type not in ['white', 'OU']:
        raise ValueError('invalid noise type, must be "white" or "OU"')
    
    for t in range(1, tot_steps):
        if blown_up:
            x_trajectory[t:, :] = np.nan
            break
        try:
            dW = np.random.normal(0, np.sqrt(dt), d)
            noise = B@dW
            dx = -A@x*dt + noise
            x += dx
            if np.any(~np.isfinite(x)):
                x_trajectory[t:, :] = np.nan
                blown_up = True
                break
            x_trajectory[t] = x
            if noise_type == 'white':
                x_trajectory[t, d_eta:] = noise[:d_eta]
        except:
            x_trajectory[t:, :] = np.nan
            blown_up = True
            break
    
    if variable == 'voltage':
        v_fp = W@r_fp + h_tot
        x_plot = x_trajectory + np.concatenate([v_fp, v_fp])
    elif variable == 'rate':
        x_plot = x_trajectory + np.concatenate([r_fp, r_fp])
    else:
        raise ValueError('invalid variable type, must be "voltage" or "rate"')
    
    return x_trajectory, x_plot, blown_up

def sim_global(r_fp, init_perturb_mag, W, h_tot, TAU, variable, T_sim=10, dt=1e-4):
    d = W.shape[0]
    tot_steps = int(T_sim/dt)
    x_trajectory = np.zeros((tot_steps, d))
    init_perturb = init_perturb_mag*np.random.uniform(-1, 1, size=d)
    tau_inv = np.linalg.inv(TAU)
    blown_up = False

    if variable == 'voltage':
        x_init = W@r_fp + h_tot + init_perturb
        x = x_init.copy()
        x_trajectory[0] = x
        for i in range(1, tot_steps):
            if blown_up:
                x_trajectory[i:, :] = np.nan
                break
            try:
                f_v = np.maximum(0, x) ** 2
                dx_dt = tau_inv @ (-x + W@f_v + h_tot)
                x += dx_dt * dt
                if np.any(~np.isfinite(x)):
                    x_trajectory[i:, :] = np.nan
                    blown_up = True
                    break 
                x_trajectory[i] = x
            except:
                x_trajectory[i:, :] = np.nan
                blown_up = True
                break

    elif variable == 'rate':
        x_init = r_fp + init_perturb
        x = x_init.copy()
        x_trajectory[0] = x
        for i in range(1, tot_steps):
            if blown_up:
                x_trajectory[i:, :] = np.nan
                break
            try:
                z = W@x + h_tot
                f_z = np.maximum(0, z) ** 2
                dx_dt = tau_inv @ (-x + f_z)
                x += dx_dt * dt
                if np.any(~np.isfinite(x)):
                    x_trajectory[i:, :] = np.nan
                    blown_up = True
                    break
                x_trajectory[i] = x
            except:
                x_trajectory[i:, :] = np.nan
                blown_up = True
                break
    else:
        raise ValueError('invalid variable type, must be "voltage" or "rate"')

    return x_trajectory, blown_up

def solve_lyapunov(A, B):
    Q = B @ B.T
    return scipy.linalg.solve_continuous_lyapunov(A, Q) # note Q positive definite 

def rate_cov_transform(cov, W):
    d = cov.shape[0]
    d_x = d // 2
    W_inv = np.linalg.inv(W)
    cov_vv = cov[:d_x, :d_x]
    cov_vn = (cov[:d_x, d_x:] + cov[d_x:, :d_x].T)/2
    cov_nn = cov[d_x:, d_x:]
    r_cov = W_inv @ (cov_vv - cov_vn - cov_vn.T + cov_nn) @ W_inv.T
    return r_cov

def estimate_online_covariance(x_array, W, A, B, variable): 
    tot_steps, d = x_array.shape
    d_x = d // 2 

    if variable == 'voltage':
        mean = np.zeros(d_x)
        cov = np.zeros((d_x, d_x))
        cov_all = np.zeros((tot_steps, d_x, d_x))
        errors = np.zeros((tot_steps, d_x, d_x))

        lyapunov_cov = solve_lyapunov(A, B)[:d_x, :d_x]

        for i in range(tot_steps):
            x_current = x_array[i, :d_x] 
            if i == 0:
                mean = x_current
                cov = np.zeros((d_x, d_x))
            else:
                diff = x_current - mean
                mean = mean + (diff) / (i) # mean + (x_current - mean) / (i + 1)

                cov = cov + (np.outer(diff, diff) - cov) / i # (i - 1) / i * cov + np.outer(diff, diff) / (i + 1)

            cov_all[i] = cov
            errors[i] = np.abs(cov - lyapunov_cov)

        # overall_cov = np.cov(x_array[:, :d_x].T)  # memory issues with calculating at once..
    
    elif variable == 'rate':
        mean = np.zeros(d)
        cov = np.zeros((d, d))
        cov_all = np.zeros((tot_steps, d, d))
        errors = np.zeros((tot_steps, d, d))

        lyapunov_cov = solve_lyapunov(A, B)

        for i in range(tot_steps):
            x_current = x_array[i, :d] 
            if i == 0:
                mean = x_current
                cov = np.zeros((d, d))
            else:
                diff = x_current - mean
                mean = mean + (diff) / (i) # mean + (x_current - mean) / (i + 1)

                cov = cov + (np.outer(diff, diff) - cov) / i # (i - 1) / i * cov + np.outer(diff, diff) / (i + 1)
        
            cov_all[i] = cov
            errors[i] = np.abs(cov - lyapunov_cov)

        # overall_cov = np.cov(x_array.T) # memory issues with calculating at once..

        cov_all = np.array([rate_cov_transform(cov, W) for cov in cov_all])
        lyapunov_cov = rate_cov_transform(lyapunov_cov, W)
        errors = errors[:, :d_x, :d_x]

    else:
        raise ValueError('invalid variable type, must be "voltage" or "rate"')

    return cov_all, lyapunov_cov, errors

def plot_variance_subplots(r_fp, init_perturb_mag, W, h_tot, dt, T_sim, TAU, tau_noise, noise_levels, sigma_E, variable, noise_type, contrast, savefig=False):
    A, _, init_perturb = get_params(r_fp, W, TAU, tau_noise, sigma_E, init_perturb_mag, dt, variable, noise_type) 
    noise_range = np.linspace(noise_levels[0], noise_levels[-1], 7)
    d = A.shape[0]
    d_x = d // 2
    total_std = np.zeros((d_x, len(noise_levels), len(noise_range), len(noise_range)))
    total_std_lyapunov = np.zeros((d_x, len(noise_levels),len(noise_range),len(noise_range)))
    labels = ['E','P','S','V']

    for Y, sigma_P in enumerate(noise_levels):
        for x, sigma_V in enumerate(noise_range):
            for y, sigma_S in enumerate(noise_range):
                SIGMA_noise = np.diag([sigma_E, sigma_P, sigma_S, sigma_V])
                if variable == 'voltage' and noise_type == 'white':
                    B = np.block([[SIGMA_noise, np.zeros((d_x,d_x))], [np.zeros((d_x,d_x)), np.zeros((d_x,d_x))]]) 
                elif variable == 'voltage' and noise_type == 'OU':
                    B = np.block([[np.zeros((d_x,d_x)), np.zeros((d_x,d_x))], [np.zeros((d_x,d_x)), np.sqrt(2/tau_noise)*SIGMA_noise]])
                elif variable == 'rate' and noise_type == 'OU':
                    B = np.block([[np.zeros((d_x,d_x)), np.sqrt(2/tau_noise)*SIGMA_noise], [np.zeros((d_x,d_x)), np.sqrt(2/tau_noise)*SIGMA_noise]])
                else:
                    raise ValueError('invalid variable/noise type, must be "voltage + white/OU" or "rate + OU"')
                x_trajectory, _, _ = sim_local(r_fp, init_perturb, W, h_tot, A, B, variable, noise_type, T_sim, dt)
                covs, lyapunov_cov, _ = estimate_online_covariance(x_trajectory, W, A, B, variable)
                total_std[:,Y,x,y] = np.sqrt(np.diagonal(covs[-1]))
                total_std_lyapunov[:,Y,x,y] = np.sqrt(np.diagonal(lyapunov_cov))

    vmin = np.min([np.min(total_std), np.min(total_std_lyapunov)])
    vmax = np.max([np.max(total_std), np.max(total_std_lyapunov)])
    fig, axes = plt.subplots(len(noise_levels), len(labels), figsize=(12, 8)) ###
    title = f'{variable} + {noise_type} noise, contrast = {contrast*100}%'
    fig.suptitle(title, fontsize=16)
    cmap = plt.get_cmap('YlOrRd')

    for X, X_str in enumerate(labels):
        X_label = f'$\sigma_{{{X_str}}}$'
        for Y, sigma_P in enumerate(noise_levels):
            Y_label = f'$\Sigma_{{S}}$, and $\Sigma_{{P}} = {sigma_P:.2f}$' 

            std_XY = total_std[X, Y, :, :]
            std_XY_lyapunov = total_std_lyapunov[X,Y,:,:]

            ax = axes[len(noise_levels) - 1 - Y, X]  # axes in reverse row order
            cax = ax.imshow(std_XY, interpolation='nearest', origin='lower', cmap='YlOrRd', aspect='equal', vmin=vmin, vmax=vmax, alpha=0.6) # alpha=0.6

            lyapunov_min = np.min(std_XY_lyapunov)
            lyapunov_max = np.max(std_XY_lyapunov)
            contour_levels = np.linspace(lyapunov_min, lyapunov_max, len(noise_range) + 2)[1:-1]  # exclude min and max
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            contour_colors = cmap(norm(contour_levels))
            contours = ax.contour(std_XY_lyapunov, levels=contour_levels, colors=contour_colors, linewidths=0.9)

            if Y == len(noise_levels) - 1:
                ax.set_title(X_label, pad=10)
            if Y == 0:
                ax.set_xlabel(f'$\Sigma_{{V}}$')
            if X == 0:
                ax.set_ylabel(Y_label)

            ax.set_xticks([-0.5, len(noise_range)/2 - 0.5, len(noise_range) - 0.5])  # ticks at the corners
            ax.set_xticklabels(['0.0', '0.2', '0.4'])
            ax.set_yticks([-0.5, len(noise_range)/2 - 0.5, len(noise_range) - 0.5])
            ax.set_yticklabels(['0.0', '0.2', '0.4'])

            if Y != 0:
                ax.set_xticks([])
            if X != 0:
                ax.set_yticks([])
            if X == len(labels) - 1:
                cbar = fig.colorbar(cax, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

                cbar.set_ticks([vmin,  vmax]) 
                cbar.set_ticklabels([f'{vmin:.1f}', f'{vmax:.1f}']) 

    plt.tight_layout()
    plt.show()

    if savefig:
        os.makedirs("../figures", exist_ok=True)
        datet = datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = f"../figures/cov_subplots_sigE{sigma_E}_{variable}_{noise_type}_noise_{datet}.pdf"
        plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300, format='pdf')
        print(f"figure saved at {save_path}")

    return fig

def compute_cov_grid(params_list, r_fp_list, contrast, init_perturb_mag, TAU, tau_noise, sigma_noise, dt, T_sim, noise_lims, n_levels, variable, noise_type):
    n_fits = len(params_list)
    d_x = params_list[0][0].shape[0]
    d = 2*d_x
    noise_levels = np.linspace(noise_lims[0], noise_lims[1], n_levels)
    
    variances_sim = np.zeros((n_fits, d_x, d_x, len(noise_levels)))
    variances_lyap = np.zeros((n_fits, d_x, d_x, len(noise_levels)))
    var_labels = ['E', 'P', 'S', 'V']
    
    for k, ((W, h1, h0), r_fp) in enumerate(zip(params_list, r_fp_list)):
        h_tot = h1 * contrast + h0
        A, _, init_perturb = get_params(r_fp, W, TAU, tau_noise, sigma_noise, init_perturb_mag, dt, variable, noise_type)
        
        for i, type_in in enumerate(var_labels):
            for j, type_out in enumerate(var_labels):
                vars_sim = np.zeros(len(noise_levels))
                vars_lyap = np.zeros(len(noise_levels))
                
                for n, noise_level in enumerate(noise_levels):
                    SIGMA_noise = np.zeros((d_x, d_x))
                    SIGMA_noise[i,i] = noise_level
                    # SIGMA_noise = np.diag([sigma_E, sigma_P, sigma_S, sigma_V])
                    if variable == 'voltage' and noise_type == 'white':
                        B = np.block([[SIGMA_noise, np.zeros((d_x,d_x))], [np.zeros((d_x,d_x)), np.zeros((d_x,d_x))]]) # np.sqrt(2/tau_noise)*np.diag([Sigma_E, Sigma_P, Sigma_S, Sigma_V])
                    elif variable == 'voltage' and noise_type == 'OU':
                        B = np.block([[np.zeros((d_x,d_x)), np.zeros((d_x,d_x))], [np.zeros((d_x,d_x)), np.sqrt(2/tau_noise)*SIGMA_noise]])
                    elif variable == 'rate' and noise_type == 'OU':
                        B = np.block([[np.zeros((d_x,d_x)), np.sqrt(2/tau_noise)*SIGMA_noise], [np.zeros((d_x,d_x)), np.sqrt(2/tau_noise)*SIGMA_noise]])
                    else:
                        raise ValueError('invalid variable/noise type, must be "voltage + white/OU" or "rate + OU"')
                    
                    x_trajectory, _, _ = sim_local(r_fp, init_perturb, W, h_tot, A, B, variable, noise_type, T_sim, dt)
                    covs, lyapunov_cov, _ = estimate_online_covariance(x_trajectory, W, A, B, variable)
                    
                    vars_sim[n] = covs[-1, j, j]
                    vars_lyap[n] = lyapunov_cov[j, j]
                
                variances_sim[k, i, j, :] = vars_sim
                variances_lyap[k, i, j, :] = vars_lyap
    
    return variances_sim, variances_lyap 

def get_params_fits_list(top_params, top_fits, c_range, contrast=None, top_k=20, random=False):
    if contrast:
        c_ind = np.where(c_range == contrast)[0][0]
    params_list, r_fp_list = [], []
    if random:
        random_indices = np.random.choice(len(top_params), top_k, replace=False)
        for i in random_indices:
            params_list.append((top_params[i][0], top_params[i][1], top_params[i][2]))
            if contrast:
                r_fp_list.append(top_fits[i][:, c_ind])
            else:
                r_fp_list.append(top_fits[i])
    else: 
        for i in range(top_k): 
            params_list.append((top_params[i][0], top_params[i][1], top_params[i][2]))
            if contrast:
                r_fp_list.append(top_fits[i][:, c_ind])
            else:
                r_fp_list.append(top_fits[i])
    return params_list, r_fp_list

#########

def evaluate_stability(x, W, h1, h0, c, tau):
    _, _, J, _, _ = get_ss_linearise(x, W, h1, h0, c, tau)
    eigenvalues = np.linalg.eigvals(J)
    max_ev = np.max(np.real(eigenvalues))
    return max_ev

def check_fixed_points(W, h1, h0, c, tau, r_fp_list=None, bounds=None, num_initial=25, atol=0):
   fixed_points = []
   
   if r_fp_list is not None:
       for idx, x0 in enumerate(r_fp_list):
           x0 = np.array(x0, dtype=np.float64)
           max_eig = evaluate_stability(x0, W, h1, h0, c, tau)
           if max_eig < atol:
               fixed_points.append(x0)
               print(f"point {idx}: {x0} is a stable fixed point")
           else:
               print(f"point {idx}: {x0} is an unstable fixed point")

   return np.array(fixed_points, dtype=np.float64) if fixed_points else np.array([], dtype=np.float64)

def system_derivative(x, W, h1, h0, c, tau):
    tau_inv = np.linalg.inv(tau)
    z = W @ x + h1 * c + h0
    f_z = np.maximum(0, z) ** 2
    return tau_inv @ (-x + f_z)

def plot_single_phase_plane(W, h1, h0, h_ext, contrast, tau, x_pop, y_pop, x_lims, y_lims, k, r_fp_list=None, resolution=50, savefig=False):

    h0_tot = h0 + h_ext
    fs = 22
    plt.rcParams.update({'font.size': fs})
    var_labels = ['E', 'P', 'S', 'V']
    i, j = var_labels.index(y_pop), var_labels.index(x_pop)
    
    base_state = np.zeros(len(W)) if r_fp_list is None else r_fp_list[0]
    
    fig = plt.figure(figsize=(8, 7))
    gs = plt.GridSpec(2, 1, height_ratios=[1, 6], hspace=0.1)
    legend_ax = plt.subplot(gs[0])
    ax = plt.subplot(gs[1])
    
    legend_ax.set_frame_on(False)
    legend_ax.axis('off')
    ax.set_aspect('equal', adjustable='box')
    
    x = np.linspace(x_lims[0], x_lims[1], resolution)
    y = np.linspace(y_lims[0], y_lims[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    U = np.zeros((resolution, resolution))
    V = np.zeros((resolution, resolution))
    magnitude = np.zeros((resolution, resolution))
    x_nullcline = np.zeros((resolution, resolution))
    y_nullcline = np.zeros((resolution, resolution))
    
    for ii in range(resolution):
        for jj in range(resolution):
            state = base_state.copy()
            state[j] = X[ii, jj]
            state[i] = Y[ii, jj]
            derivative = system_derivative(state, W, h1, h0_tot, contrast, tau)
            U[ii, jj] = derivative[j]
            V[ii, jj] = derivative[i]
            magnitude[ii, jj] = np.sqrt(derivative[j]**2 + derivative[i]**2)
            x_nullcline[ii, jj] = derivative[j]
            y_nullcline[ii, jj] = derivative[i]
    
    scalar_field = ax.pcolormesh(X, Y, magnitude, cmap='viridis')
    
    speed = np.sqrt(U**2 + V**2)
    lw = 3 * speed / speed.max()
    strm = ax.streamplot(X, Y, U, V, color='white', density=0.8,
                        linewidth=lw, arrowsize=1)
    strm.lines.set_alpha(1)
    if hasattr(strm, 'arrows'):
        strm.arrows.set_alpha(1)
    
    ax.contour(X, Y, x_nullcline, levels=[0], colors='r', linestyles=':', alpha=0.7)
    ax.contour(X, Y, y_nullcline, levels=[0], colors='b', linestyles=':', alpha=0.7)
    
    fps = check_fixed_points(W, h1, h0_tot, contrast, tau, r_fp_list=r_fp_list)
    if len(fps) > 0:
        ax.plot(fps[:, j], fps[:, i], 'rx', markersize=10)
    
    other_dims = [k for k in range(len(W)) if k not in [i, j]]
    other_x0 = [base_state[k] for k in other_dims]
    
    ax.set_xlabel(f'$r_{{{x_pop}}}$', fontsize=fs)
    ax.set_ylabel(f'$r_{{{y_pop}}}$', fontsize=fs)
    
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    ax.set_xticks(np.arange(x_lims[0], x_lims[1] + 1, 2))
    ax.set_yticks(np.arange(y_lims[0], y_lims[1] + 1, 2))
    
    # Apply fontsize to tick labels
    ax.tick_params(axis='both', which='major', labelsize=fs)
    
    legend_elements = [
        plt.Line2D([0], [0], marker='x', color='r', label='stable FPs',
                  linestyle='none', markersize=10),
        plt.Line2D([0], [0], color='r', linestyle=':', 
                  label=f'$dr_{{{x_pop}}}/dt = 0$'),
        plt.Line2D([0], [0], color='b', linestyle=':', 
                  label=f'$dr_{{{y_pop}}}/dt = 0$')
    ]
    legend_ax.legend(handles=legend_elements, loc='center', ncol=3, 
                    bbox_to_anchor=(0.47, 0.43), frameon=False, fontsize=20)
    
    cbar = plt.colorbar(scalar_field, ax=ax, label='speed')
    cbar.ax.tick_params(labelsize=fs) 
    cbar.set_label('speed', fontsize=fs) 
    
    plt.tight_layout()

    if savefig:
        os.makedirs("../figures", exist_ok=True)
        datet = datetime.now().strftime("%y%m%d_%H%M%S")
        h_ext_str = re.sub(r"\D", "", str(h_ext))
        save_path = f"../figures/phase_plane_k{k}_c{contrast}_hext{h_ext_str}_{datet}.pdf"
        plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300, format='pdf')
        print(f"figure saved at {save_path}")  

    return fig, ax

def plot_upper_triangular_phase_planes(W, h1, h0, h_ext, contrast, tau, x_lims, y_lims, k, 
                                  variable_labels=None, r_fp_list=None, resolution=50, 
                                  savefig=False, show_fps=True):
    if variable_labels is None:
        variable_labels = ['E', 'P', 'S', 'V']
    
    h0_tot = h0 + h_ext
    num_vars = len(variable_labels)
    base_state = np.zeros(len(W)) if r_fp_list is None else r_fp_list[0]
    plt.rcParams.update({'font.size': 14})
    
    fig = plt.figure(figsize=(9, 9))
    
    gs = plt.GridSpec(num_vars, num_vars, figure=fig, left=0.05)
    gs.update(wspace=0.1, hspace=0.1)
    
    axes = {}
    all_magnitudes = []
    
    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            x = np.linspace(x_lims[0], x_lims[1], resolution)
            y = np.linspace(y_lims[0], y_lims[1], resolution)
            X, Y = np.meshgrid(x, y)
            
            magnitude = np.zeros_like(X)
            for ii in range(resolution):
                for jj in range(resolution):
                    state = base_state.copy()
                    state[j] = X[ii, jj]
                    state[i] = Y[ii, jj]
                    derivative = system_derivative(state, W, h1, h0_tot, contrast, tau)
                    magnitude[ii, jj] = np.sqrt(derivative[i]**2 + derivative[j]**2)
            
            all_magnitudes.append(magnitude)
    
    global_min = min(mag.min() for mag in all_magnitudes)
    global_max = max(mag.max() for mag in all_magnitudes)
    
    xticks = [x_lims[0], np.mean(x_lims), x_lims[1]]
    yticks = [y_lims[0], np.mean(y_lims), y_lims[1]]
    
    magnitude_idx = 0
    for i in range(num_vars):
        for j in range(num_vars):
            if j > i: 
                ax = fig.add_subplot(gs[i, j])
                ax.set_aspect('equal', adjustable='box')
                axes[(i, j)] = ax
                
                x_idx, y_idx = j, i  # j is x-axis, i is y-axis
                
                x = np.linspace(x_lims[0], x_lims[1], resolution)
                y = np.linspace(y_lims[0], y_lims[1], resolution)
                X, Y = np.meshgrid(x, y)
                
                magnitude = all_magnitudes[magnitude_idx]
                magnitude_idx += 1
                
                U = np.zeros_like(X)
                V = np.zeros_like(Y)
                x_nullcline = np.zeros_like(X)
                y_nullcline = np.zeros_like(Y)
                
                for ii in range(resolution):
                    for jj in range(resolution):
                        state = base_state.copy()
                        state[x_idx] = X[ii, jj]
                        state[y_idx] = Y[ii, jj]
                        derivative = system_derivative(state, W, h1, h0_tot, contrast, tau)
                        U[ii, jj] = derivative[x_idx]
                        V[ii, jj] = derivative[y_idx]
                        x_nullcline[ii, jj] = derivative[x_idx]
                        y_nullcline[ii, jj] = derivative[y_idx]
                
                scalar_field = ax.pcolormesh(X, Y, magnitude, 
                                          vmin=global_min, vmax=global_max,
                                          cmap='viridis', shading='nearest')
                
                speed = np.sqrt(U**2 + V**2)
                lw = 2 * speed / speed.max()
                strm = ax.streamplot(X, Y, U, V, color='white', density=1,
                                   linewidth=lw, arrowsize=0.5)
                strm.lines.set_alpha(0.7)
                if hasattr(strm, 'arrows'):
                    strm.arrows.set_alpha(0.7)
                
                ax.contour(X, Y, x_nullcline, levels=[0], colors='r', linestyles=':', alpha=0.7)
                ax.contour(X, Y, y_nullcline, levels=[0], colors='b', linestyles=':', alpha=0.7)
                
                if show_fps:
                    fps = check_fixed_points(W, h1, h0_tot, contrast, tau, r_fp_list=r_fp_list)
                    if len(fps) > 0:
                        ax.plot(fps[:, x_idx], fps[:, y_idx], "rx", markersize=8)
                
                ax.set_xlim(x_lims)
                ax.set_ylim(y_lims)
                ax.set_xticks(xticks)
                ax.set_yticks(yticks)
                
                for spine in ax.spines.values():
                    spine.set_visible(False)
                
                if i == 0:
                    ax.xaxis.set_ticks_position('top')
                    ax.xaxis.set_label_position('top')
                    ax.set_xticklabels([f'{val:.1f}' for val in xticks], fontsize=10)
                    ax.set_xlabel(f'{variable_labels[j]}', fontsize=12)
                else:
                    ax.set_xticklabels([])
                
                if j == i + 1:
                    ax.set_yticklabels([f'{val:.1f}' for val in yticks], fontsize=10)
                    ax.set_ylabel(f'{variable_labels[i]}', fontsize=12)
                else:
                    ax.set_yticklabels([])
            
            else: 
                ax = fig.add_subplot(gs[i, j])
                ax.axis('off')
                axes[(i, j)] = ax
    
    legend_elements = [
        plt.Line2D([0], [0], marker='x', color='r', label='stable FPs',
                  linestyle='none', markersize=8),
        plt.Line2D([0], [0], color='r', linestyle=':', 
                  label=r'$dr_{x}/dt = 0$'),
        plt.Line2D([0], [0], color='b', linestyle=':', 
                  label=r'$dr_{y}/dt = 0$')
    ]
    
    fig.legend(handles=legend_elements, loc='center left', 
              bbox_to_anchor=(1.02, 0.65), frameon=False, fontsize=12)
    
    cbar_ax = fig.add_axes([0.93, 0.3, 0.03, 0.6])
    cbar = plt.colorbar(scalar_field, cax=cbar_ax)
    cbar.ax.set_title('speed', fontsize=14, pad=10)
    cbar.locator = matplotlib.ticker.MaxNLocator(nbins=5)
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=12)
    
    title_str = f'phase plane analysis, {int(contrast*100)}% contrast'
    fig.suptitle(title_str, fontsize=16, y=0.98)
    
    plt.tight_layout() # rect=[0.4, 0.4, 0.9, 0.9]
    
    if savefig:
        os.makedirs("../figures", exist_ok=True)
        datet = datetime.now().strftime("%y%m%d_%H%M%S")
        h_ext_str = re.sub(r"\D", "", str(h_ext))
        save_path = f"../figures/phase_planes_combinations_k{k}_c{contrast}_hext{h_ext_str}_{datet}.pdf"
        plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300, format='pdf')
        print(f"figure saved at {save_path}")
    
    return fig, axes