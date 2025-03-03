import numpy as np
import scipy
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import copy
import logging
import itertools
import sys
import os
from datetime import datetime
import ast
import re

from .fitting_functions import linearise, get_scond_0, get_scond_1, get_scond_2, get_scond_3, get_ss_linearise, get_ss
from .simulation_functions import sim_global, solve_lyapunov, get_params, rate_cov_transform
    
def get_gain_stab_cov_heatmap_combined(fit_c, W, h1, h0, contrast, pop1_idx, pop2_idx, lims, points, TAU,
                                       sigma_noise, tau_noise, variable, noise_type,
                                       init_perturb_mag=0, dt=1e-4):

    pop_range = np.linspace(lims[0], lims[1], points)
    
    stability_arr = np.full((points, points, 4), np.nan)
    gain_arr = np.full((points, points, 4), np.nan)
    real_ev_arr = np.full((points, points), np.nan)
    imag_ev_arr = np.full((points, points), np.nan)
    rates_arr = np.full((points, points, 4), np.nan)
    cov_arr = np.full((points, points, 4), np.nan)

    for i, h1_val in enumerate(pop_range):
        fit_c_1 = fit_c  
        for j, h2_val in enumerate(pop_range):
            try:
                h_perturb = np.zeros(4)
                h_perturb[pop1_idx] = h1_val
                h_perturb[pop2_idx] = h2_val
                h0_tot = h0 + h_perturb

                f_grad, X, J, main_fp, main_fp_error = get_ss_linearise(fit_c_1, W, h1, h0_tot, contrast, TAU)

                gain_arr[i, j, :] = f_grad
                rates_arr[i, j, :] = main_fp

                try:
                    stability_arr[i, j, 0] = get_scond_0(J)
                    stability_arr[i, j, 1] = get_scond_1(J)
                    stability_arr[i, j, 2] = get_scond_2(TAU, f_grad, X, J)
                    stability_arr[i, j, 3] = get_scond_3(J)
                except Exception:
                    stability_arr[i, j, 3] = np.nan

                try:
                    eigenvals = np.linalg.eigvals(J)
                    real_ev_arr[i, j] = np.max(np.real(eigenvals))
                    imag_ev_arr[i, j] = np.max(np.imag(eigenvals))
                except Exception:
                    real_ev_arr[i, j] = np.nan
                    imag_ev_arr[i, j] = np.nan

                A, B, _ = get_params(main_fp, W, TAU, tau_noise, sigma_noise,
                                     init_perturb_mag, dt, variable, noise_type)
                if real_ev_arr[i, j] > 0:
                    cov_arr[i, j] = np.full(4, np.nan)
                else:
                    lyap_cov = solve_lyapunov(A, B)
                    if variable == 'rate':
                        lyap_cov = rate_cov_transform(lyap_cov, W)
                    cov_arr[i, j] = np.diag(lyap_cov[:4, :4])
                
            except Exception as e:
                cov_arr[i, j] = np.nan
                print(f"Error at i={i}, j={j}: {str(e)}")
                print('gain=', f_grad, 'J_min=', J.min(), 'J_max=', J.max(), 'det=', np.linalg.det(J))
                continue

            fit_c_1 = main_fp

    return stability_arr, gain_arr, real_ev_arr, imag_ev_arr, rates_arr, cov_arr

def plot_single_heatmap_any(fit_c, params, lims, points, pop1, pop2, contrast,
                         sigma_noise, TAU, tau_noise, variable, noise_type, k, 
                         param_plot='stability', pop_plot='E', hatched=False, savefig=False):

    W, h1, h0 = params
    pop_names = ['E', 'PV', 'SOM', 'VIP']
    pop_labels = ['E', 'P', 'S', 'V']
    mask_color = '#EE3535'

    pop1_idx, pop2_idx, pop_plot_index = pop_labels.index(pop1), pop_labels.index(pop2), pop_labels.index(pop_plot)
    stability_arr, gain_arr, real_ev_arr, imag_ev_arr, rates_arr, cov_arr = get_gain_stab_cov_heatmap_combined(fit_c, W, h1, h0, contrast,
                                           pop1_idx, pop2_idx, lims, points, TAU,
                                           sigma_noise, tau_noise, variable, noise_type)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    x = np.linspace(lims[0], lims[1], points)
    y = np.linspace(lims[0], lims[1], points)
    X, Y = np.meshgrid(x, y)

    if param_plot == 'stability':
        data = real_ev_arr 
        cmap = copy.copy(plt.cm.RdBu_r)
        cmap.set_bad(mask_color, 1.0) 
        plot_title = 'stability'
        cbar_label = 'Re(λ)'
    elif param_plot == 'rate':
        data = rates_arr[:, :, pop_plot_index]
        cmap = copy.copy(plt.cm.Greys)
        cmap.set_bad(mask_color, 1.0) 
        plot_title = f'{pop_plot}' + ' rate'
        cbar_label = rf'$r_{{{pop_plot}}}$'
    elif param_plot == 'gain':
        data = gain_arr[:, :, pop_plot_index]
        cmap = copy.copy(plt.cm.viridis)
        cmap.set_bad(mask_color, 1.0)
        plot_title = f'{pop_plot}' + ' gain'
        cbar_label = 'gain'
    elif param_plot == 'variance':
        data = cov_arr[:, :, pop_plot_index]
        cmap = copy.copy(plt.cm.inferno)
        cmap.set_bad(mask_color, 1.0) 
        plot_title = f'{pop_plot}' + ' variance'
        cbar_label = rf'$\Sigma_{{{pop_plot}}}$'
    else:
        raise ValueError("`param` must be one of 'stability', 'rate', 'gain', or 'variance'.")

    data = np.ma.masked_invalid(data)
    
    valid_data = real_ev_arr.size > 0 and not np.all(np.isnan(real_ev_arr))
    
    unstable_mask = None
    if valid_data:
        unstable_mask = np.zeros_like(real_ev_arr, dtype=bool)
        valid_mask = ~np.isnan(real_ev_arr)
        unstable_mask[valid_mask] = real_ev_arr[valid_mask] > 0
    
    plot_data = data.copy()
    if hatched and valid_data and unstable_mask is not None and np.any(unstable_mask):
        plot_data = np.ma.array(plot_data, mask=np.ma.getmask(plot_data) | unstable_mask)

    if param_plot == 'stability': 
        if data.count() > 0:             
            percentiles = np.percentile(data.compressed(), [5, 95])             
            abs_max = np.max(np.abs(percentiles))         
        else:             
            abs_max = 1         
        norm = matplotlib.colors.TwoSlopeNorm(vmin=-abs_max, vcenter=0., vmax=abs_max)
        vmin, vmax = -abs_max, abs_max  
    else:         
        if data.count() > 0:             
            vmin = np.percentile(data.compressed(), 5)             
            vmax = np.percentile(data.compressed(), 95)         
        else:             
            raise ValueError("no valid data to plot")
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)      

    if data.count() > 0:
        if hatched and valid_data and unstable_mask is not None and np.any(unstable_mask):
            c = ax.pcolormesh(X, Y, plot_data, cmap=cmap, norm=norm, shading='auto')
        else:
            c = ax.pcolormesh(X, Y, data, cmap=cmap, norm=norm, shading='auto')
    else:         
        ax.pcolormesh(X, Y, np.zeros_like(data), cmap=plt.cm.Greys, shading='auto')         
        ax.set_facecolor(mask_color) 
        return fig
    
    boundary_mask = None
    if valid_data and unstable_mask is not None:
        boundary_mask = np.zeros_like(unstable_mask)
        
        for i in range(unstable_mask.shape[0]):
            for j in range(unstable_mask.shape[1]-1):
                if unstable_mask[i, j] != unstable_mask[i, j+1]:
                    boundary_mask[i, j] = True
                    boundary_mask[i, j+1] = True
        
        for i in range(unstable_mask.shape[0]-1):
            for j in range(unstable_mask.shape[1]):
                if unstable_mask[i, j] != unstable_mask[i+1, j]:
                    boundary_mask[i, j] = True
                    boundary_mask[i+1, j] = True
    
    if hatched and valid_data and unstable_mask is not None and np.any(unstable_mask):
        unstable_region = np.ma.masked_where(~unstable_mask, np.ones_like(unstable_mask))
        ax.contourf(X, Y, unstable_region, hatches=['\\'], alpha=0.8, colors=mask_color, extend='both')
    
    stability_conditions = [
        ('red', 'trace'),
        ('green', 'trace cubed'),
        ('blue', 'paradoxical'),
        ('orange', 'det')
    ]
    
    legend_elements = []
    for condition_idx, (color, label) in enumerate(stability_conditions):
        crossings_x, crossings_y = interp_zero_crossings(stability_arr[:, :, condition_idx], x, y)
        if len(crossings_x) > 0 and len(crossings_y) > 0:
            boundary_points_x = []
            boundary_points_y = []
            inner_points_x = []
            inner_points_y = []
            
            if boundary_mask is not None:
                boundary_flat = boundary_mask.flatten()
                grid_points = np.column_stack((X.flatten(), Y.flatten()))
                
                for px, py in zip(crossings_x, crossings_y):
                    point = np.array([px, py])
                    distances = np.sum((grid_points - point)**2, axis=1)
                    nearest_idx = np.argmin(distances)
                    
                    if boundary_flat[nearest_idx]:
                        boundary_points_x.append(px)
                        boundary_points_y.append(py)
                    else:
                        inner_points_x.append(px)
                        inner_points_y.append(py)
            else:
                boundary_points_x = crossings_x
                boundary_points_y = crossings_y
            
            if len(boundary_points_x) > 0:
                ax.scatter(boundary_points_x, boundary_points_y, color=color, edgecolors='none',
                          s=15, linewidth=0, alpha=0.65)
                
                legend_elements.append(plt.Line2D([0], [0], color=color, markersize=10, label=label))
            
            if not hatched and len(inner_points_x) > 0:
                ax.scatter(inner_points_x, inner_points_y, color=color, edgecolors='none',
                          s=15, linewidth=0, alpha=0.65)
        
    if param_plot == 'stability':         
        vmax_lim = max(abs(vmin), abs(vmax)) 
        cbar = plt.colorbar(c, ax=ax, fraction=0.1, pad=0.05, shrink=0.86, ticks=[-vmax_lim, 0, vmax_lim])
        cbar.ax.set_yticklabels([f'{-vmax_lim:.2g}', '0', f'{vmax_lim:.2g}'])     
    else:         
        cbar = plt.colorbar(c, ax=ax, fraction=0.1, pad=0.05, shrink=0.86)         
        cbar.locator = matplotlib.ticker.MaxNLocator(nbins=5)         
        cbar.update_ticks()     

    cbar.ax.set_title(cbar_label, pad=10)

    ax.set_aspect('equal')
    ax.set_xlabel(pop_names[pop2_idx])
    ax.set_ylabel(pop_names[pop1_idx])
    
    min_tick = int(np.ceil(lims[0]))
    max_tick = int(np.floor(lims[1]))
    xticks = np.arange(min_tick, max_tick + 1)
    yticks = np.arange(min_tick, max_tick + 1)
    
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f'{x:.0f}' for x in xticks])
    ax.set_yticklabels([f'{y:.0f}' for y in yticks])

    fig.legend(handles=legend_elements, loc='upper right',
               bbox_to_anchor=(1.22, 0.6), ncol=1, fontsize=14)

    title_str = f'{plot_title} analysis, {pop_names[pop1_idx]} vs {pop_names[pop2_idx]}, contrast={int(contrast*100)}%' 
    if param_plot == 'variance':
        title_str += f', noise_mag={sigma_noise}, noise_type={noise_type}'
    plt.suptitle(title_str, y=0.96)

    plt.tight_layout()

    if savefig:
        os.makedirs("../figures", exist_ok=True)
        datet = datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = f"../figures/{param_plot}_heatmap_k{k}_c{contrast}_{pop_labels[pop1_idx]}v{pop_labels[pop2_idx]}_plot{pop_plot}_{datet}.pdf"
        plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300, format='pdf')
        print(f"figure saved at {save_path}")

    return fig

def plot_heatmap_grid_any(params_list, fits_list, c_range, lims, points, TAU, 
                          pop1, pop2, sigma_noise, tau_noise, variable, 
                          noise_type, param_plot='stability', pop_plot='E', hatched=False, savefig=False):

    n_params = len(params_list)
    n_contrasts = len(c_range)
    pop_names = ['E', 'PV', 'SOM', 'VIP']
    pop_labels = ['E', 'P', 'S', 'V']
    mask_color = '#EE3535'
    pop1_idx, pop2_idx, pop_plot_index = pop_labels.index(pop1), pop_labels.index(pop2), pop_labels.index(pop_plot)
    
    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(n_params, n_contrasts, 
                             figsize=(4 * n_contrasts, 4 * n_params),
                             squeeze=False)
    
    stability_conditions = [
        ('red', 'trace'), 
        ('green', 'trace cubed'), 
        ('blue', 'paradoxical'), 
        ('orange', 'det')
    ]
    
    data_arrays = []
    stability_arrays = []
    real_ev_arrays = [] 
    vmin_global = float('inf')
    vmax_global = float('-inf')
    
    for i, (W, h1, h0) in enumerate(params_list):
        row_data = []
        row_stability = []
        row_real_ev = []  
        fit_i = fits_list[i]
        for j, contrast in enumerate(c_range):
            fit_c = fit_i[:, j]
            stability_arr, gain_arr, real_ev_arr, imag_ev_arr, rates_arr, cov_arr = \
                get_gain_stab_cov_heatmap_combined(fit_c, W, h1, h0, contrast, 
                                                   pop1_idx, pop2_idx, lims, points, 
                                                   TAU, sigma_noise, tau_noise, variable, noise_type)
            if param_plot == 'stability':
                data_arr = real_ev_arr
            elif param_plot == 'rate':
                data_arr = rates_arr[:, :, pop_plot_index]
            elif param_plot == 'gain':
                data_arr = gain_arr[:, :, pop_plot_index]
            elif param_plot == 'variance':
                data_arr = cov_arr[:, :, pop_plot_index]
            else:
                raise ValueError("`param` must be one of 'stability', 'rate', 'gain', or 'variance'.")
            
            row_data.append(data_arr)
            row_stability.append(stability_arr)
            row_real_ev.append(real_ev_arr) 
            
            masked_data = np.ma.masked_invalid(data_arr)
            if masked_data.count() > 0:
                vmin_global = min(vmin_global, np.percentile(masked_data.compressed(), 5))
                vmax_global = max(vmax_global, np.percentile(masked_data.compressed(), 95))
        data_arrays.append(row_data)
        stability_arrays.append(row_stability)
        real_ev_arrays.append(row_real_ev) 
    
    if param_plot == 'stability':
        cmap = copy.copy(plt.cm.RdBu_r)
        cmap.set_bad(mask_color, 1.0)
        abs_max_global = max(abs(vmin_global), abs(vmax_global))
        norm = matplotlib.colors.TwoSlopeNorm(vmin=-abs_max_global, vcenter=0., vmax=abs_max_global)
        plot_title = 'stability'
        cbar_label = 'Re(λ)'
    elif param_plot == 'rate':
        cmap = copy.copy(plt.cm.Greys)
        cmap.set_bad(mask_color, 1.0)
        norm = matplotlib.colors.Normalize(vmin=vmin_global, vmax=vmax_global)
        plot_title = f'{pop_plot} rate'
        cbar_label = rf'$r_{{{pop_plot}}}$'
    elif param_plot == 'gain':
        cmap = copy.copy(plt.cm.viridis)
        cmap.set_bad(mask_color, 1.0)
        norm = matplotlib.colors.Normalize(vmin=vmin_global, vmax=vmax_global)
        plot_title = f'{pop_plot} gain'
        cbar_label = 'gain'
    elif param_plot == 'variance':
        cmap = copy.copy(plt.cm.inferno)
        cmap.set_bad(mask_color, 1.0)
        norm = matplotlib.colors.Normalize(vmin=vmin_global, vmax=vmax_global)
        plot_title = f'{pop_plot} variance'
        cbar_label = rf'$\Sigma_{{{pop_plot}}}$'
    
    x = np.linspace(lims[0], lims[1], points)
    y = np.linspace(lims[0], lims[1], points)
    X, Y = np.meshgrid(x, y)
    
    for i in range(n_params):
        for j, contrast in enumerate(c_range):
            ax = axes[i, j]
            data_arr = np.ma.masked_invalid(data_arrays[i][j])
            stability_arr = stability_arrays[i][j]
            real_ev_arr = real_ev_arrays[i][j]
            
            have_stability_data = real_ev_arr.size > 0 and not np.all(np.isnan(real_ev_arr))
            
            unstable_mask = None
            if have_stability_data:
                unstable_mask = np.zeros_like(real_ev_arr, dtype=bool)
                valid_mask = ~np.isnan(real_ev_arr)
                unstable_mask[valid_mask] = real_ev_arr[valid_mask] > 0
            
            plot_data = data_arr.copy()
            if hatched and have_stability_data and unstable_mask is not None and np.any(unstable_mask):
                plot_data = np.ma.array(plot_data, mask=np.ma.getmask(plot_data) | unstable_mask)
                c = ax.pcolormesh(X, Y, plot_data, cmap=cmap, norm=norm, shading='auto')
            else:
                if data_arr.count() > 0:
                    c = ax.pcolormesh(X, Y, data_arr, cmap=cmap, norm=norm, shading='auto')
                else:
                    c = ax.pcolormesh(X, Y, np.zeros_like(data_arr), cmap=plt.cm.Greys, shading='auto')
                    ax.set_facecolor(mask_color)
            
            boundary_mask = None
            if have_stability_data and unstable_mask is not None:
                boundary_mask = np.zeros_like(unstable_mask)
                
                for ii in range(unstable_mask.shape[0]):
                    for jj in range(unstable_mask.shape[1]-1):
                        if unstable_mask[ii, jj] != unstable_mask[ii, jj+1]:
                            boundary_mask[ii, jj] = True
                            boundary_mask[ii, jj+1] = True
                
                for ii in range(unstable_mask.shape[0]-1):
                    for jj in range(unstable_mask.shape[1]):
                        if unstable_mask[ii, jj] != unstable_mask[ii+1, jj]:
                            boundary_mask[ii, jj] = True
                            boundary_mask[ii+1, jj] = True
            
            if hatched and have_stability_data and unstable_mask is not None and np.any(unstable_mask):
                unstable_region = np.ma.masked_where(~unstable_mask, np.ones_like(unstable_mask))
                ax.contourf(X, Y, unstable_region, hatches=['\\'], alpha=0.8, colors=mask_color, extend='both')
            
            for condition_idx, (color, label) in enumerate(stability_conditions):
                crossings_x, crossings_y = interp_zero_crossings(stability_arr[:, :, condition_idx], x, y)
                if len(crossings_x) > 0 and len(crossings_y) > 0:
                    boundary_points_x = []
                    boundary_points_y = []
                    inner_points_x = []
                    inner_points_y = []
                    
                    if boundary_mask is not None:
                        boundary_flat = boundary_mask.flatten()
                        grid_points = np.column_stack((X.flatten(), Y.flatten()))
                        
                        for px, py in zip(crossings_x, crossings_y):
                            point = np.array([px, py])
                            distances = np.sum((grid_points - point)**2, axis=1)
                            nearest_idx = np.argmin(distances)
                            
                            if boundary_flat[nearest_idx]:
                                boundary_points_x.append(px)
                                boundary_points_y.append(py)
                            else:
                                inner_points_x.append(px)
                                inner_points_y.append(py)
                    else:
                        boundary_points_x = crossings_x
                        boundary_points_y = crossings_y
                    
                    if len(boundary_points_x) > 0:
                        ax.scatter(boundary_points_x, boundary_points_y, color=color, edgecolors='none',
                                  s=20, linewidth=0, alpha=0.65)
                    
                    if not hatched and len(inner_points_x) > 0:
                        ax.scatter(inner_points_x, inner_points_y, color=color, edgecolors='none',
                                  s=20, linewidth=0, alpha=0.65)
            
            ax.set_xlim(lims[0], lims[1])
            ax.set_ylim(lims[0], lims[1])
            
            xticks = [lims[0], np.mean(lims), lims[1]]
            yticks = [lims[0], np.mean(lims), lims[1]]
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_xticklabels([f'{x:.1f}' for x in xticks], fontsize=12)
            ax.set_yticklabels([f'{y:.1f}' for y in yticks], fontsize=12)
            ax.set_aspect('equal')

            for spine in ax.spines.values():
                spine.set_visible(False)
            if i != n_params - 1:
                ax.set_xticks([])
            if j != 0:
                ax.set_yticks([])
            
            if i == n_params - 1:
                ax.set_xlabel(pop_names[pop2_idx], fontsize=14)
            if j == 0:
                ax.set_ylabel(pop_names[pop1_idx], fontsize=14)
            
            ax.set_title(f'Fit {i + 1}, {int(contrast * 100)}% contrast', fontsize=14)
    
    legend_elements = [
        plt.Line2D([0], [0], color=color, markerfacecolor=color, markersize=10, label=label, lw=2)
        for color, label in stability_conditions
    ]
    fig.legend(handles=legend_elements, loc='upper right',
               bbox_to_anchor=(1.12, 0.6), ncol=1, fontsize=14)
    
    title_str = f'{plot_title} analysis across fits/contrasts, {pop_labels[pop1_idx]} vs {pop_labels[pop2_idx]}'
    fig.suptitle(title_str, y=1.01, fontsize=16)
    
    cbar_ax = fig.add_axes([1.0, 0.15, 0.02, 0.7])
    if param_plot == 'stability':
        abs_max_global = max(abs(vmin_global), abs(vmax_global))
        cbar = fig.colorbar(c, cax=cbar_ax, ticks=[-abs_max_global, 0, abs_max_global])
        cbar.ax.set_yticklabels([f'{-abs_max_global:.2g}', '0', f'{abs_max_global:.2g}'])
    else:
        cbar = fig.colorbar(c, cax=cbar_ax)
        cbar.locator = matplotlib.ticker.MaxNLocator(nbins=5)
        cbar.update_ticks()
    cbar.ax.set_title(cbar_label, pad=10)
    cbar.ax.tick_params(labelsize=14)

    plt.tight_layout()

    if savefig:
        os.makedirs("../figures", exist_ok=True)
        datet = datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = f"../figures/{param_plot}_heatmap_top{n_params}_all_contrasts_{pop_labels[pop1_idx]}v{pop_labels[pop1_idx]}_plot{pop_plot}_{datet}.pdf"
        plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300, format='pdf')
        print(f"figure saved at {save_path}")
        
    return fig, axes

def plot_heatmap_combinations_any(fit_c, params, contrast, lims, points, TAU, 
                                  sigma_noise, tau_noise, variable, noise_type, k,
                                  param_plot='stability', pop_plot='E', savefig=False, hatched=False):

    pop_names = ['E', 'PV', 'SOM', 'VIP']
    row_pops = pop_names[:-1]
    col_pops = pop_names[1:]
    pop_labels = ['E', 'P', 'S', 'V']
    pop_indices = {'E': 0, 'PV': 1, 'SOM': 2, 'VIP': 3} 
    mask_color = '#EE3535'
    
    W, h1, h0 = params
    pop_plot_index = pop_indices[pop_plot]

    fig = plt.figure(figsize=(10, 9)) 
    
    data_dict = {}       
    stability_dict = {}  
    real_ev_dict = {}    
    all_data_vals = []   
    vmin_global = float('inf')
    vmax_global = float('-inf')
    
    stability_conditions = [
        ('red', 'trace'),
        ('green', 'trace cubed'),
        ('blue', 'paradoxical'),
        ('orange', 'det')
    ]
    
    conditions_present = {condition[1]: False for condition in stability_conditions}

    for i, row_pop in enumerate(row_pops):
        for j, col_pop in enumerate(col_pops[i:]):
            stability_arr, gain_arr, real_ev_arr, imag_ev_arr, rates_arr, cov_arr = get_gain_stab_cov_heatmap_combined(
                    fit_c, W, h1, h0, contrast, 
                    pop_indices[row_pop], pop_indices[col_pop],
                    lims, points, TAU, sigma_noise, tau_noise, variable, noise_type
                )
            
            if param_plot == 'stability':
                data = real_ev_arr
            elif param_plot == 'rate':
                data = rates_arr[:, :, pop_plot_index]
            elif param_plot == 'gain':
                data = gain_arr[:, :, pop_plot_index]
            elif param_plot == 'variance':
                data = cov_arr[:, :, pop_plot_index]
            else:
                raise ValueError("`param` must be one of 'stability', 'rate', 'gain', or 'variance'.")
            
            data = np.ma.masked_invalid(data)
            data_dict[(row_pop, col_pop)] = data
            stability_dict[(row_pop, col_pop)] = stability_arr
            real_ev_dict[(row_pop, col_pop)] = real_ev_arr
            
            if data.count() > 0:
                all_data_vals.extend(data.compressed())
                vmin_global = min(vmin_global, np.percentile(data.compressed(), 5))
                vmax_global = max(vmax_global, np.percentile(data.compressed(), 95))
    
    if param_plot == 'stability':
        cmap = copy.copy(plt.cm.RdBu_r)
        cmap.set_bad(mask_color, 1.0)
        plot_title = 'stability'
        data_label = 'Re(λ)'
        abs_max_global = max(abs(vmin_global), abs(vmax_global))
        norm = matplotlib.colors.TwoSlopeNorm(vmin=-abs_max_global, vcenter=0., vmax=abs_max_global)
    elif param_plot == 'rate':
        cmap = copy.copy(plt.cm.Greys)
        cmap.set_bad(mask_color, 1.0)
        plot_title = f'{pop_plot} rate'
        data_label = rf'$r_{{{pop_plot}}}$'
        norm = matplotlib.colors.Normalize(vmin=vmin_global, vmax=vmax_global)
    elif param_plot == 'gain':
        cmap = copy.copy(plt.cm.viridis)
        cmap.set_bad(mask_color, 1.0)
        plot_title = f'{pop_plot} gain'
        data_label = 'gain'
        norm = matplotlib.colors.Normalize(vmin=vmin_global, vmax=vmax_global)
    elif param_plot == 'variance':
        cmap = copy.copy(plt.cm.inferno)
        cmap.set_bad(mask_color, 1.0)
        plot_title = f'{pop_plot} variance'
        data_label = rf'$\Sigma_{{{pop_plot}}}$'
        norm = matplotlib.colors.Normalize(vmin=vmin_global, vmax=vmax_global)
    
    x = np.linspace(lims[0], lims[1], points)
    y = np.linspace(lims[0], lims[1], points)
    X, Y = np.meshgrid(x, y)

    xticks = [lims[0], np.mean(lims), lims[1]]
    yticks = [lims[0], np.mean(lims), lims[1]]
    
    for i, row_pop in enumerate(row_pops):
        for j, col_pop in enumerate(col_pops[i:]):
            actual_j = i + j 
            base_plot_num = (i * 3) + (actual_j) + 1 

            ax = fig.add_subplot(3, 3, base_plot_num)

            data = data_dict[(row_pop, col_pop)]
            stability_arr = stability_dict[(row_pop, col_pop)]
            real_ev_arr = real_ev_dict[(row_pop, col_pop)]
            
            valid_data = real_ev_arr.size > 0 and not np.all(np.isnan(real_ev_arr))
            
            unstable_mask = None
            if valid_data:
                unstable_mask = np.zeros_like(real_ev_arr, dtype=bool)
                valid_mask = ~np.isnan(real_ev_arr)
                unstable_mask[valid_mask] = real_ev_arr[valid_mask] > 0
            
            plot_data = data.copy()
            if hatched and valid_data and unstable_mask is not None and np.any(unstable_mask):
                plot_data = np.ma.array(plot_data, mask=np.ma.getmask(plot_data) | unstable_mask)
                c = ax.pcolormesh(X, Y, plot_data, cmap=cmap, norm=norm, shading='nearest')
            else:
                c = ax.pcolormesh(X, Y, data, cmap=cmap, norm=norm, shading='nearest')
            
            boundary_mask = None
            if valid_data and unstable_mask is not None:
                boundary_mask = np.zeros_like(unstable_mask)
                for ii in range(unstable_mask.shape[0]):
                    for jj in range(unstable_mask.shape[1]-1):
                        if unstable_mask[ii, jj] != unstable_mask[ii, jj+1]:
                            boundary_mask[ii, jj] = True
                            boundary_mask[ii, jj+1] = True
                
                for ii in range(unstable_mask.shape[0]-1):
                    for jj in range(unstable_mask.shape[1]):
                        if unstable_mask[ii, jj] != unstable_mask[ii+1, jj]:
                            boundary_mask[ii, jj] = True
                            boundary_mask[ii+1, jj] = True
            
            if hatched and valid_data and unstable_mask is not None and np.any(unstable_mask):
                unstable_region = np.ma.masked_where(~unstable_mask, np.ones_like(unstable_mask))
                ax.contourf(X, Y, unstable_region, hatches=['\\'], alpha=0.8, colors=mask_color, extend='both')
            
            for condition_idx, (color, label) in enumerate(stability_conditions):
                crossings_x, crossings_y = interp_zero_crossings(stability_arr[:, :, condition_idx], x, y)
                if len(crossings_x) > 0 and len(crossings_y) > 0:
                    boundary_points_x = []
                    boundary_points_y = []
                    inner_points_x = []
                    inner_points_y = []
                    
                    if boundary_mask is not None:
                        boundary_flat = boundary_mask.flatten()
                        grid_points = np.column_stack((X.flatten(), Y.flatten()))
                        
                        for px, py in zip(crossings_x, crossings_y):
                            point = np.array([px, py])
                            distances = np.sum((grid_points - point)**2, axis=1)
                            nearest_idx = np.argmin(distances)
                            if boundary_flat[nearest_idx]:
                                boundary_points_x.append(px)
                                boundary_points_y.append(py)
                            else:
                                inner_points_x.append(px)
                                inner_points_y.append(py)
                    else:
                        boundary_points_x = crossings_x
                        boundary_points_y = crossings_y
                    
                    if len(boundary_points_x) > 0:
                        ax.scatter(boundary_points_x, boundary_points_y, color=color, s=2, alpha=0.65)
                        conditions_present[label] = True
                    
                    if not hatched and len(inner_points_x) > 0:
                        ax.scatter(inner_points_x, inner_points_y, color=color, s=2, alpha=0.65)
                        if not conditions_present[label]:
                            conditions_present[label] = True
            
            ax.set_xlim(lims[0], lims[1])
            ax.set_ylim(lims[0], lims[1])
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_aspect('equal', adjustable='box')
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            if i == 0:
                ax.xaxis.set_ticks_position('top')
                ax.xaxis.set_label_position('top')
                ax.set_xticklabels([f'{val:.3g}' for val in xticks], fontsize=10)
                ax.set_xlabel(f'{col_pop}', fontsize=12)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_yticklabels([f'{val:.3g}' for val in yticks], fontsize=10)
                ax.set_ylabel(row_pop, fontsize=12)
            else:
                ax.set_yticklabels([])
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.6])
    
    if param_plot == 'stability':
        cbar = fig.colorbar(c, cax=cbar_ax, ticks=[-abs_max_global, 0, abs_max_global])
        cbar.ax.set_yticklabels([f'{-abs_max_global:.2g}', '0', f'{abs_max_global:.2g}'])
    else:
        cbar = fig.colorbar(c, cax=cbar_ax)
        cbar.locator = matplotlib.ticker.MaxNLocator(nbins=5)
        cbar.update_ticks()
    
    cbar.ax.set_title(data_label, fontsize=14, pad=10)
    cbar.ax.tick_params(labelsize=12)
    
    legend_elements = [
        plt.Line2D([0], [0], color=color, markerfacecolor=color, markersize=6, label=label, lw=2)
        for color, label in stability_conditions
        if conditions_present[label]
    ]
    
    if legend_elements:
        fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.01, 0.5))
    
    title_str = f'{plot_title} analysis across combinations, {int(contrast*100)}% contrast'
    
    fig.suptitle(title_str, fontsize=16, y=0.92)
    plt.tight_layout(rect=[0, 0, 0.9, 0.93])

    if savefig:
        os.makedirs("../figures", exist_ok=True)
        datet = datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = f"../figures/{param_plot}_heatmap_combinations_k{k}_c{contrast}_plot{pop_plot}_{datet}.pdf"
        plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300, format='pdf')
        print(f"figure saved at {save_path}")

    return fig

def plot_stab_rate_gain_var_single(fit_c, params, lims, points, pop1, pop2, contrast, sigma_noise, TAU, tau_noise, variable, noise_type, k, pop_plot='E', hatched=False, savefig=False):
    W, h1, h0 = params
    pop_names = ['E','PV','SOM','VIP']
    pop_labels = ['E', 'P', 'S', 'V']
    mask_color = '#EE3535'
    pop1_idx, pop2_idx, pop_plot_index = pop_labels.index(pop1), pop_labels.index(pop2), pop_labels.index(pop_plot)

    stability_arr, gain_arr, real_ev_arr, imag_ev_arr, rates_arr, cov_arr = get_gain_stab_cov_heatmap_combined(fit_c, W, h1, h0, contrast, pop1_idx, pop2_idx, lims, points, TAU, sigma_noise, tau_noise, variable, noise_type)

    plt.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(12, 12))
    grid = plt.GridSpec(2, 2, hspace=0.08, wspace=0.2)
    ax_re = fig.add_subplot(grid[0, 0]) # stability
    ax_rate = fig.add_subplot(grid[0, 1]) # rate
    ax_gain = fig.add_subplot(grid[1, 0]) # E gain
    ax_cov = fig.add_subplot(grid[1, 1]) # E variance
    
    x = np.linspace(lims[0], lims[1], points)
    y = np.linspace(lims[0], lims[1], points)
    X, Y = np.meshgrid(x, y)
    
    cmap_re = copy.copy(plt.cm.RdBu_r)
    cmap_re.set_bad(mask_color, 1.0)
    cmap_rate = copy.copy(plt.cm.Greys)
    cmap_rate.set_bad(mask_color, 1.0)
    cmap_gain = copy.copy(plt.cm.viridis)
    cmap_gain.set_bad(mask_color, 1.0)
    cmap_cov = copy.copy(plt.cm.inferno)
    cmap_cov.set_bad(mask_color, 1.0)

    have_stability_data = real_ev_arr.size > 0 and not np.all(np.isnan(real_ev_arr))
    
    unstable_mask = None
    if have_stability_data:
        unstable_mask = np.zeros_like(real_ev_arr, dtype=bool)
        valid_mask = ~np.isnan(real_ev_arr)
        unstable_mask[valid_mask] = real_ev_arr[valid_mask] > 0

    plots = [
        (ax_re, real_ev_arr, cmap_re, 'max eigenvalue', 'Re(λ)'),
        (ax_rate, rates_arr[:,:,pop_plot_index], cmap_rate, 'E rate', r'$r_E$'),
        (ax_gain, gain_arr[:,:,pop_plot_index], cmap_gain, 'E gain', 'gain'),
        (ax_cov, cov_arr[:,:,pop_plot_index], cmap_cov, 'E variance', r'$\Sigma_{E}$')
    ]

    for ax, data, cmap, title, cbar_label in plots:
        data = np.ma.masked_invalid(data)
        
        if title == 'max eigenvalue':
            abs_max = np.max(abs(np.percentile(data.compressed(), [5, 95])))
            norm = matplotlib.colors.TwoSlopeNorm(vmin=-abs_max, vcenter=0., vmax=abs_max)
        else:
            vmin = np.percentile(data.compressed(), 5)
            vmax = np.percentile(data.compressed(), 95)
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        
        plot_data = data.copy()
        if hatched and have_stability_data and unstable_mask is not None and np.any(unstable_mask):
            plot_data = np.ma.array(plot_data, mask=np.ma.getmask(plot_data) | unstable_mask)
        
        if data.count() > 0:
            c = ax.pcolormesh(X, Y, plot_data, cmap=cmap, norm=norm, shading='auto')
        else:
            ax.pcolormesh(X, Y, np.zeros_like(data), cmap=plt.cm.Greys, shading='auto')
            ax.set_facecolor(mask_color)
        
        if hatched and have_stability_data and unstable_mask is not None and np.any(unstable_mask):
            unstable_region = np.ma.masked_where(~unstable_mask, np.ones_like(unstable_mask))
            ax.contourf(X, Y, unstable_region, hatches=['\\'], alpha=0.8, colors=mask_color, extend='both')
        
        cbar = plt.colorbar(c, ax=ax, fraction=0.05, pad=0.08, shrink = 0.83)
        cbar.ax.set_title(cbar_label, pad=10)
        
        ax.set_title(title)
        
        ax.set_aspect('equal')
        ax.set_xlabel(f'{pop_names[pop2_idx]}')
        ax.set_ylabel(f'{pop_names[pop1_idx]}')
        
        ax.set_xticks([lims[0], lims[1]])
        ax.set_yticks([lims[0], lims[1]])
        ax.set_xticklabels([f'{val:.0f}' for val in [lims[0], lims[1]]])
        ax.set_yticklabels([f'{val:.0f}' for val in [lims[0], lims[1]]])
    
    for ax in [ax_re, ax_rate, ax_gain, ax_cov]:
        for condition_idx, (color, label) in enumerate([
            ('red', 'trace'), ('green', 'trace cubed'),
            ('blue', 'paradoxical'), ('orange', 'det')]):
            crossings_x, crossings_y = interp_zero_crossings(
                stability_arr[:, :, condition_idx], x, y
            )
            
            if len(crossings_x) > 0 and len(crossings_y) > 0:
                ax.scatter(crossings_x, crossings_y, color=color, 
                          edgecolors='none', s=6, linewidth=0,
                          alpha=0.65, label=label)
    
    lines = [plt.Line2D([0], [0], color=c, lw=2) for c in ['red', 'green', 'blue', 'orange']]
    labels = ['trace', 'trace cubed', 'paradoxical', 'det']
    fig.legend(lines, labels, loc='center right', bbox_to_anchor=(1.1, 0.5))
    
    title_str = r'stability, ' + f'rate, gain and variance analysis, {pop_names[pop1_idx]} vs {pop_names[pop2_idx]} ' + f'contrast={int(contrast*100)}%, noise_mag={sigma_noise}, noise_type={noise_type}'
    plt.suptitle(title_str, y=0.94)
    
    plt.tight_layout()

    if savefig:
        os.makedirs("../figures", exist_ok=True)
        datet = datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = f"../figures/all_heatmap_k{k}_c{contrast}_plot{pop_plot}_{datet}.pdf"
        plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300, format='pdf')
        print(f"figure saved at {save_path}")
        
    return fig

def plot_stab_rate_gain_var_grid(fit_c, params, c_range, lims, points, TAU, pop1, pop2, sigma_noise, tau_noise, variable, noise_type, k, pop_plot='E', hatched=False, savefig=False):
    W, h1, h0 = params
    pop_names = ['E', 'PV', 'SOM', 'VIP']
    pop_labels = ['E', 'P', 'S', 'V']
    n_cols = len(c_range)
    mask_color = '#EE3535'
    
    plt.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(3*n_cols, 13)) 
    gs = plt.GridSpec(5, n_cols+1, height_ratios=[1, 10, 10, 10, 10], width_ratios=[*[1]*n_cols, 0.05])
    ax_legend = fig.add_subplot(gs[0, :n_cols])
    
    all_re_vals = []
    all_rate_vals = []
    all_gain_vals = []
    all_cov_vals = []
    
    data_storage = []
    pop1_idx, pop2_idx, pop_plot_index = pop_labels.index(pop1), pop_labels.index(pop2), pop_labels.index(pop_plot)
    for col_idx, contrast in enumerate(c_range):
        stability_arr, gain_arr, real_ev_arr, imag_ev_arr, rates_arr, cov_arr = get_gain_stab_cov_heatmap_combined(fit_c, W, h1, h0, contrast, pop1_idx, pop2_idx, lims, points, TAU, sigma_noise, tau_noise, variable, noise_type)
        
        real_ev_arr = np.ma.masked_invalid(real_ev_arr)
        rates_arr = np.ma.masked_invalid(rates_arr[:, :, pop_plot_index])
        gain_arr = np.ma.masked_invalid(gain_arr[:, :, pop_plot_index])
        cov_arr = np.ma.masked_invalid(cov_arr[:, :, pop_plot_index])
        
        data_storage.append((stability_arr, real_ev_arr, rates_arr, gain_arr, cov_arr))
        
        all_re_vals.extend(real_ev_arr.compressed())
        all_rate_vals.extend(rates_arr.compressed())
        all_gain_vals.extend(gain_arr.compressed())
        all_cov_vals.extend(cov_arr.compressed())
    
    re_abs_max = np.percentile(np.abs(all_re_vals), 95)
    re_norm = matplotlib.colors.TwoSlopeNorm(vmin=-re_abs_max, vcenter=0., vmax=re_abs_max)
    
    rate_min, rate_max = np.percentile(all_rate_vals, [5, 95])
    rate_norm = matplotlib.colors.Normalize(vmin=rate_min, vmax=rate_max)
    
    gain_min, gain_max = np.percentile(all_gain_vals, [5, 95])
    gain_norm = matplotlib.colors.Normalize(vmin=gain_min, vmax=gain_max)
    
    cov_min, cov_max = np.percentile(all_cov_vals, [5, 95])
    cov_norm = matplotlib.colors.Normalize(vmin=cov_min, vmax=cov_max)
    
    cmap_re = copy.copy(plt.cm.RdBu_r)
    cmap_re.set_bad(mask_color, 1.0)
    cmap_rate = copy.copy(plt.cm.Greys)
    cmap_rate.set_bad(mask_color, 1.0)
    cmap_gain = copy.copy(plt.cm.viridis)
    cmap_gain.set_bad(mask_color, 1.0)
    cmap_cov = copy.copy(plt.cm.viridis)
    cmap_cov.set_bad(mask_color, 1.0)
    
    legend_elements = []
    for color, label in [('red', 'trace'), ('green', 'trace cubed'),
                        ('blue', 'paradoxical'), ('orange', 'det')]:
        legend_elements.append(plt.Line2D([0], [0], color=color,
                                        markerfacecolor=color, markersize=10,
                                        label=label, lw=2))
    
    ax_legend.legend(handles=legend_elements, loc='upper left', ncol=4, bbox_to_anchor=(0.0, 1.02))
    ax_legend.axis('off')
    
    row_types = [
        (real_ev_arr, cmap_re, re_norm, 'max eigenvalue', 'Re(λ)'),
        (rates_arr, cmap_rate, rate_norm, f'{pop_plot} rate', rf'$r_{{{pop_plot}}}$'),
        (gain_arr, cmap_gain, gain_norm, f'{pop_plot} gain', 'gain'),
        (cov_arr, cmap_cov, cov_norm, f'{pop_plot} variance', rf'$\Sigma_{{{pop_plot}}}$')
    ]
    
    for col_idx, (stability_arr, re_eigs, rates, gains, cov) in enumerate(data_storage):
        x = np.linspace(lims[0], lims[1], points)
        y = np.linspace(lims[0], lims[1], points)
        X, Y = np.meshgrid(x, y)
        
        have_stability_data = re_eigs.size > 0 and not np.all(np.isnan(re_eigs))
        unstable_mask = None
        if have_stability_data:
            unstable_mask = np.zeros_like(re_eigs, dtype=bool)
            valid_mask = ~np.isnan(re_eigs)
            unstable_mask[valid_mask] = re_eigs[valid_mask] > 0
        
        row_data = [re_eigs, rates, gains, cov]
        for row_idx, (data_type, cmap, norm, title, cbar_label) in enumerate(row_types):
            ax = fig.add_subplot(gs[row_idx+1, col_idx])
            data = row_data[row_idx]
            plot_data = data.copy()

            if hatched and have_stability_data and unstable_mask is not None and np.any(unstable_mask):
                plot_data = np.ma.array(plot_data, mask=np.ma.getmask(plot_data) | unstable_mask)
            if data.count() > 0:
                c = ax.pcolormesh(X, Y, plot_data, cmap=cmap, norm=norm, shading='auto')
            else:
                c = ax.pcolormesh(X, Y, np.zeros_like(data), cmap=plt.cm.Greys, shading='auto')
                ax.set_facecolor(mask_color)
            if hatched and have_stability_data and unstable_mask is not None and np.any(unstable_mask):
                unstable_region = np.ma.masked_where(~unstable_mask, np.ones_like(unstable_mask))
                
                ax.contourf(X, Y, unstable_region, hatches=['\\'], alpha=0.8, colors=mask_color, extend='both')
            for condition_idx, color in enumerate(['red', 'green', 'blue', 'orange']):
                crossings_x, crossings_y = interp_zero_crossings(
                    stability_arr[:, :, condition_idx], x, y
                )
                if len(crossings_x) > 0:
                    ax.scatter(crossings_x, crossings_y, color=color, 
                             edgecolors='none', s=20, alpha=0.65)
            
            if col_idx == n_cols-1:
                cax = fig.add_subplot(gs[row_idx+1, -1])
                cbar = plt.colorbar(c, cax=cax)
                cbar.ax.tick_params(labelsize=12)
                cbar.ax.set_title(cbar_label, pad=10, fontsize=12)
            
            ax.set_aspect('equal')
            
            if col_idx == 0:
                ax.set_ylabel(f'{pop_names[pop1_idx]}')
            if row_idx == 3: 
                ax.set_xlabel(f'{pop_names[pop2_idx]}')
            
            if col_idx == 0 or row_idx == 3:
                ax.set_xticks([lims[0], lims[1]])
                ax.set_yticks([lims[0], lims[1]])
                ax.set_xticklabels([f'{val:.0f}' for val in [lims[0], lims[1]]])
                ax.set_yticklabels([f'{val:.0f}' for val in [lims[0], lims[1]]])
            else:
                ax.set_xticks([])
                ax.set_yticks([])
            
            if row_idx == 0: 
                ax.set_title(f'{int(c_range[col_idx]*100)}%', pad=10)
            
    title_str = r'stability, ' + f'rate, gain and variance analysis, {pop_names[pop1_idx]} vs {pop_names[pop2_idx]} ' + f'contrast={int(contrast*100)}%, noise_mag={sigma_noise}, noise_type={noise_type}'
    plt.suptitle(title_str, y=0.97)
    
    plt.tight_layout()

    if savefig:
        os.makedirs("../figures", exist_ok=True)
        datet = datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = f"../figures/all_heatmap_k{k}_{pop_labels[pop1_idx]}v{pop_labels[pop2_idx]}_plot{pop_plot}_{datet}.pdf"
        plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300, format='pdf')
        print(f"figure saved at {save_path}")
        
    return fig

def plot_dual_param_combinations(fit_c, params, contrast, lims, points, TAU, 
                                sigma_noise, tau_noise, variable, noise_type, k,
                                param1='stability', param2='rate', pop_plot='E', hatched=False, savefig=False, ):
    pop_names = ['E', 'PV', 'SOM', 'VIP']
    row_pops = pop_names[:-1]
    col_pops = pop_names[1:]
    pop_labels = ['E', 'P', 'S', 'V']
    pop_indices = {'E': 0, 'PV': 1, 'SOM': 2, 'VIP': 3}
    mask_color = '#EE3535'
    
    pop_plot_index = pop_indices[pop_plot]
    plt.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(18, 9))
    
    stability_conditions = [
        ('red', 'trace'),
        ('green', 'trace cubed'),
        ('blue', 'paradoxical'),
        ('orange', 'det')
    ]
    
    conditions_present = {condition[1]: False for condition in stability_conditions}
    
    data_dict1 = {}    
    data_dict2 = {}     
    stability_dict = {}
    real_ev_dict = {}
    
    all_values1 = []
    all_values2 = []
    
    W, h1, h0 = params
    for i, row_pop in enumerate(row_pops):
        for j, col_pop in enumerate(col_pops[i:]):
            stability_arr, gain_arr, real_ev_arr, imag_ev_arr, rates_arr, cov_arr = get_gain_stab_cov_heatmap_combined(
                fit_c, W, h1, h0, contrast, 
                pop_indices[row_pop], pop_indices[col_pop],
                lims, points, TAU, sigma_noise, tau_noise, variable, noise_type
            )
            
            if param1 == 'stability':
                data1 = real_ev_arr
            elif param1 == 'imaginary':
                data1 = imag_ev_arr
            elif param1 == 'rate':
                data1 = rates_arr[:, :, pop_plot_index]
            elif param1 == 'gain':
                data1 = gain_arr[:, :, pop_plot_index]
            elif param1 == 'variance':
                data1 = cov_arr[:, :, pop_plot_index]
            else:
                raise ValueError(f"param1 must be one of 'stability', 'imaginary', 'rate', 'gain', or 'variance', got {param1}")

            if param2 == 'stability':
                data2 = real_ev_arr
            elif param2 == 'imaginary':
                data2 = imag_ev_arr
            elif param2 == 'rate':
                data2 = rates_arr[:, :, pop_plot_index]
            elif param2 == 'gain':
                data2 = gain_arr[:, :, pop_plot_index]
            elif param2 == 'variance':
                data2 = cov_arr[:, :, pop_plot_index]
            else:
                raise ValueError(f"param2 must be one of 'stability', 'imaginary', 'rate', 'gain', or 'variance', got {param2}")
            
            data1 = np.ma.masked_invalid(data1)
            data2 = np.ma.masked_invalid(data2)
            
            data_dict1[(row_pop, col_pop)] = data1
            data_dict2[(row_pop, col_pop)] = data2
            stability_dict[(row_pop, col_pop)] = stability_arr
            real_ev_dict[(row_pop, col_pop)] = real_ev_arr
            
            if data1.count() > 0:
                all_values1.extend(data1.compressed())
            if data2.count() > 0:
                all_values2.extend(data2.compressed())
    
    if param1 == 'stability':
        cmap1 = copy.copy(plt.cm.RdBu_r)
        cmap1.set_bad(mask_color, 1.0)
        data_label1 = 'Re(λ)'
        
        if all_values1:
            percentiles = np.percentile(all_values1, [5, 95])
            abs_max1 = np.max(np.abs(percentiles))
        else:
            abs_max1 = 1
        norm1 = matplotlib.colors.TwoSlopeNorm(vmin=-abs_max1, vcenter=0., vmax=abs_max1)
        vmin1, vmax1 = -abs_max1, abs_max1
    elif param1 == 'imaginary':
        cmap1 = copy.copy(plt.cm.Greys)
        cmap1.set_bad(mask_color, 1.0)
        data_label1 = 'Im(λ)'
        
        if all_values1:
            vmin1 = np.percentile(all_values1, 5)
            vmax1 = np.percentile(all_values1, 95)
        else:
            vmin1, vmax1 = 0, 1
        norm1 = matplotlib.colors.Normalize(vmin=vmin1, vmax=vmax1)
    elif param1 == 'rate':
        cmap1 = copy.copy(plt.cm.Greys)
        cmap1.set_bad(mask_color, 1.0)
        data_label1 = rf'$r_{{{pop_plot}}}$'
        
        if all_values1:
            vmin1 = np.percentile(all_values1, 5)
            vmax1 = np.percentile(all_values1, 95)
        else:
            vmin1, vmax1 = 0, 1
        norm1 = matplotlib.colors.Normalize(vmin=vmin1, vmax=vmax1)  
    elif param1 == 'gain':
        cmap1 = copy.copy(plt.cm.viridis)
        cmap1.set_bad(mask_color, 1.0)
        data_label1 = 'gain'
        
        if all_values1:
            vmin1 = np.percentile(all_values1, 5)
            vmax1 = np.percentile(all_values1, 95)
        else:
            vmin1, vmax1 = 0, 1
        norm1 = matplotlib.colors.Normalize(vmin=vmin1, vmax=vmax1)
    elif param1 == 'variance':
        cmap1 = copy.copy(plt.cm.inferno)
        cmap1.set_bad(mask_color, 1.0)
        data_label1 = rf'$\Sigma_{{{pop_plot}}}$'
        
        if all_values1:
            vmin1 = np.percentile(all_values1, 5)
            vmax1 = np.percentile(all_values1, 95)
        else:
            vmin1, vmax1 = 0, 1
        norm1 = matplotlib.colors.Normalize(vmin=vmin1, vmax=vmax1)
    
    if param2 == 'stability':
        cmap2 = copy.copy(plt.cm.RdBu_r)
        cmap2.set_bad(mask_color, 1.0)
        data_label2 = 'Re(λ)'
        if all_values2:
            percentiles = np.percentile(all_values2, [5, 95])
            abs_max2 = np.max(np.abs(percentiles))
        else:
            abs_max2 = 1
        norm2 = matplotlib.colors.TwoSlopeNorm(vmin=-abs_max2, vcenter=0., vmax=abs_max2)
        vmin2, vmax2 = -abs_max2, abs_max2
    
    elif param2 == 'imaginary':
        cmap2 = copy.copy(plt.cm.Greys)
        cmap2.set_bad(mask_color, 1.0)
        data_label2 = 'Im(λ)'
        if all_values2:
            vmin2 = np.percentile(all_values2, 5)
            vmax2 = np.percentile(all_values2, 95)
        else:
            vmin2, vmax2 = 0, 1
        norm2 = matplotlib.colors.Normalize(vmin=vmin2, vmax=vmax2)
        
    elif param2 == 'rate':
        cmap2 = copy.copy(plt.cm.Greys)
        cmap2.set_bad(mask_color, 1.0)
        data_label2 = rf'$r_{{{pop_plot}}}$'
        if all_values2:
            vmin2 = np.percentile(all_values2, 5)
            vmax2 = np.percentile(all_values2, 95)
        else:
            vmin2, vmax2 = 0, 1
        norm2 = matplotlib.colors.Normalize(vmin=vmin2, vmax=vmax2)
        
    elif param2 == 'gain':
        cmap2 = copy.copy(plt.cm.viridis)
        cmap2.set_bad(mask_color, 1.0)
        data_label2 = 'gain'
        if all_values2:
            vmin2 = np.percentile(all_values2, 5)
            vmax2 = np.percentile(all_values2, 95)
        else:
            vmin2, vmax2 = 0, 1
        norm2 = matplotlib.colors.Normalize(vmin=vmin2, vmax=vmax2)
        
    elif param2 == 'variance':
        cmap2 = copy.copy(plt.cm.inferno)
        cmap2.set_bad(mask_color, 1.0)
        data_label2 = rf'$\Sigma_{{{pop_plot}}}$'
        if all_values2:
            vmin2 = np.percentile(all_values2, 5)
            vmax2 = np.percentile(all_values2, 95)
        else:
            vmin2, vmax2 = 0, 1
        norm2 = matplotlib.colors.Normalize(vmin=vmin2, vmax=vmax2)
    
    x = np.linspace(lims[0], lims[1], points)
    y = np.linspace(lims[0], lims[1], points)
    X, Y = np.meshgrid(x, y)
    xticks = [lims[0], np.mean(lims), lims[1]]
    yticks = [lims[0], np.mean(lims), lims[1]]
    
    for i, row_pop in enumerate(row_pops):
        for j, col_pop in enumerate(col_pops[i:]):
            actual_j = i + j
            base_plot_num = (i * 6) + (actual_j * 2) + 1 
        
            ax1 = fig.add_subplot(3, 6, base_plot_num)
            ax2 = fig.add_subplot(3, 6, base_plot_num + 1)
            
            data1 = data_dict1[(row_pop, col_pop)]
            data2 = data_dict2[(row_pop, col_pop)]
            stability_arr = stability_dict[(row_pop, col_pop)]
            real_ev_arr = real_ev_dict[(row_pop, col_pop)]
        
            have_stability_data = real_ev_arr.size > 0 and not np.all(np.isnan(real_ev_arr))
            unstable_mask = None
            if have_stability_data:
                unstable_mask = np.zeros_like(real_ev_arr, dtype=bool)
                valid_mask = ~np.isnan(real_ev_arr)
                unstable_mask[valid_mask] = real_ev_arr[valid_mask] > 0
            
            plot_data1 = data1.copy()
            if hatched and have_stability_data and unstable_mask is not None and np.any(unstable_mask):
                plot_data1 = np.ma.array(plot_data1, mask=np.ma.getmask(plot_data1) | unstable_mask)
    
            plot_data2 = data2.copy()
            if hatched and have_stability_data and unstable_mask is not None and np.any(unstable_mask):
                plot_data2 = np.ma.array(plot_data2, mask=np.ma.getmask(plot_data2) | unstable_mask)
            
            if data1.count() > 0:
                c1 = ax1.pcolormesh(X, Y, plot_data1, cmap=cmap1, norm=norm1, shading='nearest')
            else:
                ax1.pcolormesh(X, Y, np.zeros_like(data1), cmap=plt.cm.Greys, shading='nearest')
                ax1.set_facecolor(mask_color)
                
            if data2.count() > 0:
                c2 = ax2.pcolormesh(X, Y, plot_data2, cmap=cmap2, norm=norm2, shading='nearest')
            else:
                ax2.pcolormesh(X, Y, np.zeros_like(data2), cmap=plt.cm.Greys, shading='nearest')
                ax2.set_facecolor(mask_color)
            
            if hatched and have_stability_data and unstable_mask is not None and np.any(unstable_mask):
                unstable_region = np.ma.masked_where(~unstable_mask, np.ones_like(unstable_mask))
                ax1.contourf(X, Y, unstable_region, hatches=['\\'], alpha=0.8, colors=mask_color, extend='both')
                ax2.contourf(X, Y, unstable_region, hatches=['\\'], alpha=0.8, colors=mask_color, extend='both')
            
            for condition_idx, (color, label) in enumerate(stability_conditions):
                crossings_x, crossings_y = interp_zero_crossings(stability_arr[:, :, condition_idx], x, y)
                if len(crossings_x) > 0 and len(crossings_y) > 0:
                    ax1.scatter(crossings_x, crossings_y, color=color, s=5, alpha=0.65)
                    ax2.scatter(crossings_x, crossings_y, color=color, s=5, alpha=0.65)
                    conditions_present[label] = True
            
            for ax in [ax1, ax2]:
                ax.set_xlim(lims[0], lims[1])
                ax.set_ylim(lims[0], lims[1])
                ax.set_xticks(xticks)
                ax.set_yticks(yticks)
                ax.set_aspect('equal', adjustable='box')
                for spine in ax.spines.values():
                    spine.set_visible(False)
            
            if i == 0:
                ax1.xaxis.set_ticks_position('top')
                ax2.xaxis.set_ticks_position('top')
                ax1.xaxis.set_label_position('top')
                ax2.xaxis.set_label_position('top')
                
                ax1.set_xticklabels([f'{val:.3g}' for val in xticks], fontsize=10)
                ax2.set_xticklabels([f'{val:.3g}' for val in xticks], fontsize=10)
                
                ax1.set_xlabel(f'{col_pop} ({data_label1})', fontsize=12)
                ax2.set_xlabel(f'{col_pop} ({data_label2})', fontsize=12)
            else:
                ax1.set_xticklabels([])
                ax2.set_xticklabels([])
            
            if j == 0:
                ax1.set_yticklabels([f'{val:.3g}' for val in yticks], fontsize=10)
                ax1.set_ylabel(row_pop, fontsize=12)
                ax2.set_yticklabels([])
            else:
                ax1.set_yticklabels([])
                ax2.set_yticklabels([])
    
    cbar_ax1 = fig.add_axes([1.00, 0.15, 0.01, 0.6])
    cbar_ax2 = fig.add_axes([1.04, 0.15, 0.01, 0.6])
    
    if param1 in ['stability']: #, 'imaginary']:
        cbar1 = fig.colorbar(c1, cax=cbar_ax1, ticks=[-abs_max1, 0, abs_max1])
        cbar1.ax.set_yticklabels([f'{-abs_max1:.2g}', '0', f'{abs_max1:.2g}'])
    else:
        cbar1 = fig.colorbar(c1, cax=cbar_ax1)
        cbar1.locator = matplotlib.ticker.MaxNLocator(nbins=5)
        cbar1.update_ticks()
    
    cbar1.ax.set_title(data_label1, fontsize=14, pad=10)
    cbar1.ax.tick_params(labelsize=12)
    
    if param2 in ['stability']: # , 'imaginary']:
        cbar2 = fig.colorbar(c2, cax=cbar_ax2, ticks=[-abs_max2, 0, abs_max2])
        cbar2.ax.set_yticklabels([f'{-abs_max2:.2g}', '0', f'{abs_max2:.2g}'])
    else:
        cbar2 = fig.colorbar(c2, cax=cbar_ax2)
        cbar2.locator = matplotlib.ticker.MaxNLocator(nbins=5)
        cbar2.update_ticks()
    
    cbar2.ax.set_title(data_label2, fontsize=14, pad=10)
    cbar2.ax.tick_params(labelsize=12)
    
    legend_elements = [
        plt.Line2D([0], [0], color=color, markerfacecolor=color, markersize=6, label=label, lw=2)
        for color, label in stability_conditions
        if conditions_present[label]
    ]
    
    if legend_elements: 
        fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.08, 0.5))
    
    title_str = f'{param1} and {param2} analysis across combinations, {int(contrast*100)}% contrast'
    fig.suptitle(title_str, fontsize=16, y=1.0)
    plt.tight_layout(rect=[0, 0, 0.99, 0.95])
    
    if savefig:
        os.makedirs("../figures", exist_ok=True)
        datet = datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = f"../figures/{param1}_{param2}_heatmap_combinations_k{k}_c{contrast}_plot{pop_plot}_{datet}.pdf"
        plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300, format='pdf')
        print(f"Figure saved at {save_path}")
    
    return fig

def plot_eigenvalue_paths(fit_c, params, contrast, lims, points, TAU, pop1, pop2, k, savefig=False):
    W, h1, h0 = params
    pop_names = ['E', 'PV', 'SOM', 'VIP']
    pop_labels = ['E', 'P', 'S', 'V']
    pop1_idx, pop2_idx = pop_labels.index(pop1), pop_labels.index(pop2)
    
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(14, 8))

    pop1_range = np.linspace(lims[0], lims[1], points)
    pop2_range = np.linspace(lims[0], lims[1], points)
    
    paths = {
        'pop1': {'color': 'red', 'real': [], 'imag': [], 'label': f'{pop_names[pop1_idx]}'},
        'pop2': {'color': 'blue', 'real': [], 'imag': [], 'label': f'{pop_names[pop2_idx]}'},
        'both': {'color': 'purple', 'real': [], 'imag': [], 'label': 'combined'}
    }
    
    fit_c_1 = fit_c
    for h_pop2 in pop2_range:
        h_perturb = np.zeros(4)
        h_perturb[pop2_idx] = h_pop2
        h0_tot = h0 + h_perturb
        f_grad, X, J, main_fp, main_fp_error = get_ss_linearise(fit_c_1, W, h1, h0_tot, contrast, TAU)
        eigenvalues = np.linalg.eigvals(J)
        max_idx = np.argmax(np.real(eigenvalues))
        paths['pop2']['real'].append(np.real(eigenvalues[max_idx]))
        paths['pop2']['imag'].append(np.imag(eigenvalues[max_idx]))
        fit_c_1 = main_fp
    
    fit_c_1 = fit_c
    for h_pop1 in pop1_range:
        h_perturb = np.zeros(4)
        h_perturb[pop1_idx] = h_pop1
        h0_tot = h0 + h_perturb
        f_grad, X, J, main_fp, main_fp_error = get_ss_linearise(fit_c_1, W, h1, h0_tot, contrast, TAU)
        eigenvalues = np.linalg.eigvals(J)
        max_idx = np.argmax(np.real(eigenvalues))
        paths['pop1']['real'].append(np.real(eigenvalues[max_idx]))
        paths['pop1']['imag'].append(np.imag(eigenvalues[max_idx]))
        fit_c_1 = main_fp
    
    fit_c_1 = fit_c
    for i in range(points):
        h_perturb = np.zeros(4)
        h_perturb[pop1_idx] = pop1_range[i]
        h_perturb[pop2_idx] = pop2_range[i]
        h0_tot = h0 + h_perturb
        f_grad, X, J, main_fp, main_fp_error = get_ss_linearise(fit_c_1, W, h1, h0_tot, contrast, TAU)
        eigenvalues = np.linalg.eigvals(J)
        max_idx = np.argmax(np.real(eigenvalues))
        paths['both']['real'].append(np.real(eigenvalues[max_idx]))
        paths['both']['imag'].append(np.imag(eigenvalues[max_idx]))
        fit_c_1 = main_fp
    
    all_real = []
    all_imag = []
    for path_data in paths.values():
        all_real.extend(path_data['real'])
        all_imag.extend(path_data['imag'])
        all_imag.extend([-im for im in path_data['imag']]) 
    
    x_min, x_max = min(all_real), max(all_real)
    max_abs_imag = max(abs(im) for im in all_imag)
    
    x_padding = max(0.2 * (x_max - x_min), 1) 
    x_min -= x_padding
    x_max += x_padding
    x_range = x_max - x_min
    y_max = x_range  
    
    if max_abs_imag * 1.2 > y_max:
        scaling_factor = max_abs_imag * 1.2 / y_max
        y_max = max_abs_imag * 1.2
        x_range_new = y_max
        x_padding_new = (x_range_new - (x_max - x_min)) / 2
        x_min -= x_padding_new
        x_max += x_padding_new
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-y_max, y_max)
    ax.set_aspect('equal')
    
    for path_type, path_data in paths.items():
        alphas = np.linspace(0.2, 1, points)
        
        for i in range(len(alphas) - 1):
            ax.plot(path_data['real'][i:i+2], path_data['imag'][i:i+2], 
                   color=path_data['color'], alpha=alphas[i], linewidth=2)
            ax.plot(path_data['real'][i:i+2], [-x for x in path_data['imag'][i:i+2]], 
                   color=path_data['color'], alpha=alphas[i], linewidth=2)
        
        ax.scatter(path_data['real'][0], path_data['imag'][0], 
                  color=path_data['color'], s=50, label=f'{path_data["label"]}')
        ax.scatter(path_data['real'][0], -path_data['imag'][0], 
                  color=path_data['color'], s=50)
        
        ax.scatter(path_data['real'][-1], path_data['imag'][-1], 
                  color=path_data['color'], marker='>', s=60)
        ax.scatter(path_data['real'][-1], -path_data['imag'][-1], 
                  color=path_data['color'], marker='>', s=60)
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.set_xlabel('Re', fontsize=14)
    ax.set_ylabel('Im', fontsize=14)
    ax.set_title(f'max eigenvalue path, {pop_names[pop1_idx]} vs {pop_names[pop2_idx]}, {int(contrast * 100)}% contrast', fontsize=14)
    ax.legend(fontsize=12, loc='center right', bbox_to_anchor=(1.2, 0.7))
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()

    if savefig:
        os.makedirs("../figures", exist_ok=True)
        datet = datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = f"../figures/eigenvalue_path_k{k}_c{contrast}_{pop_labels[pop1_idx]}v{pop_labels[pop2_idx]}_{datet}.pdf"
        plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300, format='pdf')
        print(f"figure saved at {save_path}")

    return fig, ax

def plot_eigenvalue_path_grid(params_list, fits_list, c_range, lims, points, TAU, pop1, pop2, savefig=False):
    n_params = len(params_list)
    n_contrasts = len(c_range)
    pop_names = ['E', 'PV', 'SOM', 'VIP']
    pop_labels = ['E', 'P', 'S', 'V']
    pop1_idx, pop2_idx = pop_labels.index(pop1), pop_labels.index(pop2)
    
    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(n_params, n_contrasts, 
                            figsize=(5 * n_contrasts, 5 * n_params),
                            squeeze=False)
    plt.subplots_adjust(wspace=0)
    
    max_real = float('-inf')
    max_imag = float('-inf')
    
    for i, (W, h1, h0) in enumerate(params_list):
        fit_i = fits_list[i]
        for j, contrast in enumerate(c_range):
            fit_c = fit_i[:, j]
            for stim_type, color, h_func in [
                (f'{pop_names[pop2_idx]}', 'blue', lambda t: np.array([0, 0, 0, t]) if pop2_idx == 3 else
                (f'{pop_names[pop2_idx]}', 'blue', lambda t: np.array([0, 0, 0, t]))),
                (f'{pop_names[pop1_idx]}', 'red', lambda t: np.array([0, 0, t, 0]) if pop1_idx == 2 else
                (f'{pop_names[pop1_idx]}', 'red', lambda t: np.array([0, 0, t, 0]))),
                ('combined', 'purple', lambda t: np.array([0, 0, t, t]))
            ]:
                real_path = []
                imag_path = []
                
                stim_range = np.linspace(lims[0], lims[1], points)
                fit_c_1 = fit_c
                for t in stim_range:
                    h_perturb = h_func(t)
                    h0_tot = h0 + h_perturb
                    f_grad, X, J, main_fp, main_fp_error = get_ss_linearise(fit_c_1, W, h1, h0_tot, contrast, TAU)
                    eigenvalues = np.linalg.eigvals(J)
                    max_idx = np.argmax(np.real(eigenvalues))
                    real_path.append(np.real(eigenvalues[max_idx]))
                    imag_path.append(np.imag(eigenvalues[max_idx]))
                    
                    fit_c_1 = main_fp

                max_real = max(max_real, max(abs(np.array(real_path))))
                max_imag = max(max_imag, max(abs(np.array(imag_path))))
    
    max_real = np.ceil(max_real * 1.1)
    max_imag = np.ceil(max_imag * 1.1)
    
    for i, (W, h1, h0) in enumerate(params_list):
        fit_i = fits_list[i]
        for j, contrast in enumerate(c_range):
            fit_c = fit_i[:, j]
            ax = axes[i, j]
            
            for stim_type, color, h_func in [
                (f'{pop_names[pop2_idx]}', 'blue', lambda t: np.array([0, 0, 0, t]) if pop2_idx == 3 else
                (f'{pop_names[pop2_idx]}', 'blue', lambda t: np.array([0, 0, 0, t]))),
                (f'{pop_names[pop1_idx]}', 'red', lambda t: np.array([0, 0, t, 0]) if pop1_idx == 2 else
                (f'{pop_names[pop1_idx]}', 'red', lambda t: np.array([0, 0, t, 0]))),
                ('combined', 'purple', lambda t: np.array([0, 0, t, t]))
            ]:
                real_path = []
                imag_path = []
                
                stim_range = np.linspace(lims[0], lims[1], points)
                fit_c_1 = fit_c
                for t in stim_range:
                    h_perturb = h_func(t)
                    h0_tot = h0 + h_perturb
                    f_grad, X, J, main_fp, main_fp_error = get_ss_linearise(fit_c_1, W, h1, h0_tot, contrast, TAU)
                    eigenvalues = np.linalg.eigvals(J)
                    max_idx = np.argmax(np.real(eigenvalues))
                    real_path.append(np.real(eigenvalues[max_idx]))
                    imag_path.append(np.imag(eigenvalues[max_idx]))

                    fit_c_1 = main_fp
                
                alphas = (np.linspace(0, 1, points)**2) * 0.9 + 0.1 
                for k in range(len(alphas) - 1):
                    ax.plot(real_path[k:k+2], imag_path[k:k+2], 
                           color=color, alpha=alphas[k], linewidth=1.5)
                    ax.plot(real_path[k:k+2], [-x for x in imag_path[k:k+2]], 
                           color=color, alpha=alphas[k], linewidth=1.5)
                
                if i == 0 and j == 0:
                    label = f'{stim_type}'
                else:
                    label = None
                
                ax.scatter(real_path[0], imag_path[0], color=color, s=40, label=label)
                ax.scatter(real_path[0], -imag_path[0], color=color, s=40)
                
                ax.scatter(real_path[-1], imag_path[-1], color=color, marker='>', s=80)
                ax.scatter(real_path[-1], -imag_path[-1], color=color, marker='>', s=80)
            
            ax.axhline(y=0, color='grey', alpha=0.7, linewidth=1)
            ax.axvline(x=0, color='grey', alpha=0.7, linewidth=1)
            ax.set_xticks([-max_real, 0, max_real])
            ax.set_yticks([-max_imag, 0, max_imag])
            
            if i != n_params - 1:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])
            if i == n_params - 1:
                ax.set_xticklabels([f'{-max_real:.0f}', '0', f'{max_real:.0f}'])
            if j == 0:
                ax.set_yticklabels([f'{-max_imag:.0f}', '0', f'{max_imag:.0f}'])
            
            ax.set_aspect('equal')
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            ax.set_xlim(-max_real, max_real)
            ax.set_ylim(-max_imag, max_imag)
            ax.grid(True, alpha=0.2)
            
            if i == n_params - 1:
                ax.set_xlabel('Re', fontsize=14)
            if j == 0:
                ax.set_ylabel('Im', fontsize=14)
            
            ax.set_title(f'fit {i + 1}, {int(contrast * 100)}% contrast', fontsize=14)
            
            if i == 0 and j == 0:
                ax.legend(fontsize=10)
    
    fig.suptitle(f'max eigenvalue paths across fits/contrasts, {pop_names[pop1_idx]} vs {pop_names[pop2_idx]}', fontsize=14)
    plt.tight_layout()

    if savefig:
        os.makedirs("../figures", exist_ok=True)
        datet = datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = f"../figures/eigenvalue_paths_top{n_params}_all_contrasts_{pop_labels[pop1_idx]}v{pop_labels[pop2_idx]}_{datet}.pdf"
        plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300, format='pdf')
        print(f"figure saved at {save_path}")

    return fig, axes

def stability_scan_analysis_and_plot(fit_c, params, contrast, init_perturb, final_perturb, points, TAU, k,
                                    plot_stability_lines=True, plot_max_ev=True, plot_det=True, 
                                    plot_trace=True, hatched=False, savefig=False):

    W, h1, h0 = params
    mask_color = '#EE3535'
    
    stability_arr = np.full((points, 4), np.nan)
    real_ev_arr = np.full(points, np.nan)
    manual_error_arr = np.full(points, np.nan)
    manual_fp_arr = np.full((points, 4), np.nan)
    poly_fp_arr = np.full((points, 4), np.nan)
    jacobian_arr = np.full((points, 4, 4), np.nan)
    gain_arr = np.full((points, 4), np.nan)
    eigenvals_real = np.full((points, 4), np.nan)
    eigenvals_imag = np.full((points, 4), np.nan)
    
    t = np.linspace(0, 1, points)
    
    fit_c_1 = fit_c
    for i, scale in enumerate(t):
        try:
            h_perturb = init_perturb + scale * (final_perturb - init_perturb)
            h0_tot = h0 + h_perturb
            
            f_grad, X, J, main_fp, main_fp_error = get_ss_linearise(fit_c_1, W, h1, h0_tot, contrast, TAU)
            
            poly_fp_arr[i, :] = main_fp
            jacobian_arr[i, :, :] = J
            gain_arr[i, :] = f_grad
            
            stability_arr[i, 0] = get_scond_0(J)
            stability_arr[i, 1] = get_scond_1(J)
            stability_arr[i, 2] = get_scond_2(TAU, f_grad, X, J)
            stability_arr[i, 3] = get_scond_3(J)
            
            eigenvals = np.linalg.eigvals(J)
            eigenvals_real[i, :] = np.real(eigenvals)
            eigenvals_imag[i, :] = np.imag(eigenvals)
            real_ev_arr[i] = np.max(np.real(eigenvals))

            fit_c_1 = main_fp
            
        except Exception as e:
            print(f"Error at scale={scale}: {str(e)}")
            continue
    
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    
    if plot_max_ev:
        ax1.plot(t, real_ev_arr, 'k-', label='max eigenvalue')
    
    det_arr = np.linalg.det(jacobian_arr) if plot_det else None
    trace_arr = np.trace(jacobian_arr, axis1=1, axis2=2) if plot_trace else None
    
    if plot_det:
        ax2.plot(t, det_arr, label="det magnitude", color="purple", linestyle="--")
    if plot_trace:
        ax2.plot(t, trace_arr, label="trace", color="magenta", linestyle="--")
    
    if any([plot_max_ev, plot_det]):
        ax2.set_yscale('symlog', linthresh=1e-3) 
        
        y_min, y_max = ax2.get_ylim()
        
        if y_min > 0:
            y_min = -1e-3
        if y_max < 0:
            y_max = 1e-3
    
        ax2.set_ylim(y_min, y_max)
        
        max_magnitude = max(abs(y_min), abs(y_max))
        log_ticks = [0] 
        for exp in range(-3, int(np.log10(max_magnitude)) + 1, 2): 
            tick = 10**exp
            if abs(tick) <= max_magnitude:
                log_ticks.extend([-tick, tick])
        log_ticks.sort()
        ax2.set_yticks(log_ticks)
        
        def format_tick(x):
            if abs(x) < 1e-10: 
                return "0"
            elif abs(x) < 1e-3 or abs(x) >= 1e4:
                return f"{x:.0e}"
            else:
                return f"{x:.1g}"
        
        ax2.set_yticklabels([format_tick(tick) for tick in log_ticks])
        ax2.tick_params(axis='y', which='both', labelsize=12)
    
    if plot_stability_lines:
        stab_colors = ['red', 'green', 'blue', 'orange']
        stab_labels = ['trace cnd', 'trace cubed cnd', 'paradoxical cnd', 'det cnd']
        
        for cond_idx in range(4):
            crossings_for_cond = []
            for i in range(len(t) - 1):
                if stability_arr[i, cond_idx] * stability_arr[i + 1, cond_idx] < 0:
                    x0, x1 = t[i], t[i + 1]
                    y0, y1 = stability_arr[i, cond_idx], stability_arr[i + 1, cond_idx]
                    crossing = x0 - y0 * (x1 - x0) / (y1 - y0)
                    crossings_for_cond.append(crossing)
            
            first_line = True
            for crossing in crossings_for_cond:
                if first_line:
                    ax1.axvline(x=crossing, color=stab_colors[cond_idx], linestyle="--", alpha=0.65,
                                label=stab_labels[cond_idx])
                    first_line = False
                else:
                    ax1.axvline(x=crossing, color=stab_colors[cond_idx], linestyle="--", alpha=0.65)
    
    if hatched: # and plot_max_ev:
        y_min, y_max = ax1.get_ylim()
        
        valid_indices = ~np.isnan(real_ev_arr)
        if np.any(valid_indices):
            t_valid = t[valid_indices]
            ev_valid = real_ev_arr[valid_indices]
            
            if len(t_valid) > 1: 
                positive_ev_regions = []
                prev_t = None
                prev_ev = None
                
                for i in range(len(t_valid)):
                    curr_t = t_valid[i]
                    curr_ev = ev_valid[i]
                    
                    if prev_ev is not None and prev_ev <= 0 and curr_ev > 0:
                        cross_t = prev_t + (0 - prev_ev) * (curr_t - prev_t) / (curr_ev - prev_ev)
                        positive_ev_regions.append([cross_t, None]) 
                    
                    elif prev_ev is not None and prev_ev > 0 and curr_ev <= 0:
                        cross_t = prev_t + (0 - prev_ev) * (curr_t - prev_t) / (curr_ev - prev_ev)
                        if positive_ev_regions and positive_ev_regions[-1][1] is None:
                            positive_ev_regions[-1][1] = cross_t 
                    
                    prev_t = curr_t
                    prev_ev = curr_ev
                
                if positive_ev_regions and positive_ev_regions[-1][1] is None:
                    positive_ev_regions[-1][1] = 1.0
                
                if ev_valid[0] > 0 and (not positive_ev_regions or positive_ev_regions[0][0] > 0):
                    positive_ev_regions.insert(0, [0.0, positive_ev_regions[0][0] if positive_ev_regions else 1.0])
                
                for start, end in positive_ev_regions:
                    ax1.axvspan(start, end, alpha=0.4, color=mask_color, hatch='\\', zorder=0)
    
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", bbox_to_anchor=(1.2, 0.7))
    
    ax1.set_xlim(0, 1)
    ax1.set_xlabel(f'relative perturb')
    ax1.set_ylabel('max eigenvalue')
    
    if any([plot_max_ev, plot_det, plot_trace]):
        metrics_label = []
        if plot_det:
            metrics_label.append("det magnitude")
        if plot_trace:
            metrics_label.append("trace")
        ax2.set_ylabel(", ".join(metrics_label) + " (log)")
        ax2.yaxis.labelpad = 10
    
    init_perturb_str = f'[{init_perturb[0]:.1g}, {init_perturb[1]:.1g}, {init_perturb[2]:.1g}, {init_perturb[3]:.1g}]'
    init_digits = re.sub(r"\D", "", init_perturb_str)
    final_perturb_str = f'[{final_perturb[0]:.1g}, {final_perturb[1]:.1g}, {final_perturb[2]:.1g}, {final_perturb[3]:.1g}]'
    final_digits = re.sub(r"\D", "", final_perturb_str)
    fig.suptitle(f'max ev, det, and trace, contrast={int(contrast * 100)}%, init_input={init_perturb_str}, final_input={final_perturb_str}', fontsize=15, y=0.96)
    
    plt.tight_layout()
    
    data_dict = {
        't': t,
        'real_ev_arr': real_ev_arr,
        'manual_error_arr': manual_error_arr,
        'stability_arr': stability_arr,
        'manual_fp_arr': manual_fp_arr,
        'poly_fp_arr': poly_fp_arr,
        'jacobian_arr': jacobian_arr,
        'eigenvals_real': eigenvals_real,
        'eigenvals_imag': eigenvals_imag,
        'gain_arr': gain_arr,
        'contrast': contrast,
        'k' : k,
        'init_perturb': init_perturb,
        'final_perturb': final_perturb
    }
    
    if savefig:
        os.makedirs("../figures", exist_ok=True)
        datet = datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = f"../figures/stability_scan_k{k}_c{contrast}_start{init_digits}_end{final_digits}_{datet}.pdf"
        plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300, format='pdf')
        print(f"figure saved at {save_path}")
    
    return fig, data_dict

def plot_rates_gains_from_data(data_dict, 
                     plot_stability_lines=True, plot_rates=True, plot_gains=True, 
                     scale='linear', hatched=False, savefig=False):

    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(8, 6))
    mask_color = '#EE3535' 
    
    t = data_dict['t']
    stability_arr = data_dict['stability_arr']
    real_ev_arr = data_dict['real_ev_arr'] 
    contrast = data_dict['contrast']
    init_perturb = data_dict['init_perturb']
    final_perturb = data_dict['final_perturb']
    k = data_dict['k']

    init_perturb_str = f'[{init_perturb[0]:.1g}, {init_perturb[1]:.1g}, {init_perturb[2]:.1g}, {init_perturb[3]:.1g}]'
    init_digits = re.sub(r"\D", "", init_perturb_str)
    final_perturb_str = f'[{final_perturb[0]:.1g}, {final_perturb[1]:.1g}, {final_perturb[2]:.1g}, {final_perturb[3]:.1g}]'
    final_digits = re.sub(r"\D", "", final_perturb_str)
    
    all_handles = []
    all_labels = []
    
    if hatched:
        valid_indices = ~np.isnan(real_ev_arr)
        if np.any(valid_indices):
            t_valid = t[valid_indices]
            ev_valid = real_ev_arr[valid_indices]
            
            if len(t_valid) > 1: 
                positive_ev_regions = []
                prev_t = None
                prev_ev = None
                
                for i in range(len(t_valid)):
                    curr_t = t_valid[i]
                    curr_ev = ev_valid[i]
                    
                    if prev_ev is not None and prev_ev <= 0 and curr_ev > 0:
                        cross_t = prev_t + (0 - prev_ev) * (curr_t - prev_t) / (curr_ev - prev_ev)
                        positive_ev_regions.append([cross_t, None]) 
                    elif prev_ev is not None and prev_ev > 0 and curr_ev <= 0:
                        cross_t = prev_t + (0 - prev_ev) * (curr_t - prev_t) / (curr_ev - prev_ev)
                        if positive_ev_regions and positive_ev_regions[-1][1] is None:
                            positive_ev_regions[-1][1] = cross_t 
                    
                    prev_t = curr_t
                    prev_ev = curr_ev
                
                if positive_ev_regions and positive_ev_regions[-1][1] is None:
                    positive_ev_regions[-1][1] = 1.0
                if ev_valid[0] > 0 and (not positive_ev_regions or positive_ev_regions[0][0] > 0):
                    positive_ev_regions.insert(0, [0.0, positive_ev_regions[0][0] if positive_ev_regions else 1.0])
                for start, end in positive_ev_regions:
                    ax.axvspan(start, end, alpha=0.4, color=mask_color, hatch='\\', zorder=0)
    
    if plot_rates:
        poly_fp_arr = data_dict['poly_fp_arr']
        pop_names = ['E', 'PV', 'SOM', 'VIP']
        rate_colors = ['#35322F', '#3FA4BC', '#EA9523', '#F2948F']
        for k in range(len(pop_names)):
            line, = ax.plot(t, poly_fp_arr[:, k], color=rate_colors[k],
                            linestyle='-', linewidth=2, alpha=0.8,
                            label=f'{pop_names[k]} rate')
            all_handles.append(line)
            all_labels.append(f'{pop_names[k]} rate')
    
    if plot_gains:
        gain_arr = data_dict['gain_arr']
        pop_names = ['E', 'PV', 'SOM', 'VIP']
        for k in range(len(pop_names)):
            line, = ax.plot(t, gain_arr[:, k], color=rate_colors[k],
                            linestyle='--', linewidth=2, alpha=0.8,
                            label=f'{pop_names[k]} gain')
            all_handles.append(line)
            all_labels.append(f'{pop_names[k]} gain')
    
    if plot_stability_lines:
        stability_colors = ['red', 'green', 'blue', 'orange']
        stability_labels = ['trace', 'trace cubed', 'paradoxical', 'det']
        for cond_idx in range(len(stability_colors)):
            zero_crossings = []
            for i in range(len(t) - 1):
                if stability_arr[i, cond_idx] * stability_arr[i + 1, cond_idx] < 0:
                    x0, x1 = t[i], t[i + 1]
                    y0, y1 = stability_arr[i, cond_idx], stability_arr[i + 1, cond_idx]
                    crossing = x0 - y0 * (x1 - x0) / (y1 - y0)
                    zero_crossings.append(crossing)
            first_line = True
            for crossing in zero_crossings:
                if first_line:
                    line = ax.axvline(x=crossing, color=stability_colors[cond_idx],
                                      linestyle='--', alpha=0.65,
                                      label=stability_labels[cond_idx])
                    all_handles.append(line)
                    all_labels.append(stability_labels[cond_idx])
                    first_line = False
                else:
                    ax.axvline(x=crossing, color=stability_colors[cond_idx],
                               linestyle='--', alpha=0.65)
    
    plotted_data = []
    if plot_rates:
        plotted_data.append(poly_fp_arr.flatten())
    if plot_gains:
        plotted_data.append(gain_arr.flatten())
    
    if len(plotted_data) > 0:
        plotted_data = np.concatenate(plotted_data)
        data_min = np.nanmin(plotted_data)
        data_max = np.nanmax(plotted_data)
    else:
        data_min, data_max = 0, 1 

    if scale == 'linear':
        ax.set_yscale('linear')
        lower_lim = min(0, data_min)
        upper_lim = max(0, data_max)
        rng = upper_lim - lower_lim
        margin = 0.05 * rng if rng != 0 else 1
        ax.set_ylim(lower_lim - margin, upper_lim + margin)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    elif scale == 'log':
        if data_min <= 0:
            positive_data = plotted_data[plotted_data > 0]
            if positive_data.size > 0:
                candidate = np.min(positive_data)
                linthresh = candidate / 2.0
            else:
                linthresh = 1e-8
            ax.set_yscale('symlog', linthresh=linthresh, linscale=1)
        else:
            ax.set_yscale('log')
        ax.set_ylim(data_min, data_max)
    else:
        raise ValueError("scale must be either 'linear' or 'log'")
    
    ax.grid(True, which='major', alpha=0.2)
    ax.grid(True, which='minor', alpha=0.1)
    ax.set_xlim(0, 1)
    ax.set_xlabel('relative perturb', fontsize=14)
    ax.set_ylabel('rates, gains', fontsize=14)
    
    plotted_items = []
    if plot_rates:
        plotted_items.append("rates")
    if plot_gains:
        plotted_items.append("gains")
    title_items = ", ".join(plotted_items) if plotted_items else ""
    fig.suptitle(f'{title_items}, contrast={int(contrast * 100)}%, init_perturb={init_perturb_str}, final_perturb={final_perturb_str}',
                 y=0.98, fontsize=14)
    
    ax.legend(all_handles, all_labels, bbox_to_anchor=(1.05, 0.8), loc='upper left')
    
    plt.tight_layout()

    if savefig:
        os.makedirs("../figures", exist_ok=True)
        datet = datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = f"../figures/scan_rates_gains_k{k}_c{contrast}_start{init_digits}_end{final_digits}_{datet}.pdf"
        plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300, format='pdf')
        print(f"figure saved at {save_path}")
    
    return fig

def plot_eigenvalue_scatter_from_data(data_dict, savefig=False):
    t = data_dict["t"]
    eigenvals_real = data_dict["eigenvals_real"]
    eigenvals_imag = data_dict["eigenvals_imag"]
    stability_arr = data_dict["stability_arr"]
    contrast = data_dict["contrast"]
    init_perturb = data_dict["init_perturb"]
    final_perturb = data_dict["final_perturb"]
    k = data_dict["k"]

    init_perturb_str = f'[{init_perturb[0]:.1g}, {init_perturb[1]:.1g}, {init_perturb[2]:.1g}, {init_perturb[3]:.1g}]'
    init_digits = re.sub(r"\D", "", init_perturb_str)
    final_perturb_str = f'[{final_perturb[0]:.1g}, {final_perturb[1]:.1g}, {final_perturb[2]:.1g}, {final_perturb[3]:.1g}]'
    final_digits = re.sub(r"\D", "", final_perturb_str)

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['#35322F', '#3FA4BC', '#EA9523', '#F2948F']
    stability_colors = ['red', 'green', 'blue', 'orange']
    stability_labels = ['trace', 'trace cubed', 'paradoxical', 'det']
    
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.5, zorder=1)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, zorder=1)
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.3, zorder=1)
    ax.add_artist(circle)
    
    max_real = np.max(np.abs(eigenvals_real))
    max_imag = np.max(np.abs(eigenvals_imag))
    max_val_re = max_real * 1.2
    max_val_im = max_imag * 1.2
    ax.set_xlim(-max_val_re, max_val_re/2)
    ax.set_ylim(-max_val_im, max_val_im)
    
    def add_end_arrow(end_point_real, end_point_imag, direction_real, direction_imag, color):
        norm = np.sqrt(direction_real**2 + direction_imag**2)
        if norm > 0:
            direction_real /= norm
            direction_imag /= norm
            arrow_scale = max_val_re * 0.03
            ax.arrow(end_point_real, end_point_imag,
                     direction_real * arrow_scale*0.6, direction_imag * arrow_scale*0.5,
                     head_width=arrow_scale * 0.7,
                     head_length=arrow_scale * 0.7,
                     fc=color, ec=color,
                     alpha=1,
                     length_includes_head=False,
                     zorder=5,
                     overhang=0
                     )
    
    max_eig_idx = np.argmax(eigenvals_real, axis=1)
    
    for i in range(eigenvals_real.shape[1]):
        alphas = np.linspace(0.2, 1, len(t))
        ax.scatter(eigenvals_real[:, i], eigenvals_imag[:, i],
                   c=colors[i], alpha=alphas, s=20,
                   zorder=2, edgecolors='none')
        ax.scatter(eigenvals_real[0, i], eigenvals_imag[0, i],
                   c=colors[i], s=60, zorder=3, edgecolors='none', label=rf'$\lambda_{{{i+1}}}$')
        if len(t) >= 2:
            direction_real = eigenvals_real[-1, i] - eigenvals_real[-2, i]
            direction_imag = eigenvals_imag[-1, i] - eigenvals_imag[-2, i]
            add_end_arrow(eigenvals_real[-1, i],
                          eigenvals_imag[-1, i],
                          direction_real,
                          direction_imag,
                          colors[i])
    
    stability_used = [False] * 4
    for j in range(len(t)-1):
        curr_max_idx = max_eig_idx[j]
        next_max_idx = max_eig_idx[j+1]
        curr_max_val = eigenvals_real[j, curr_max_idx]
        next_max_val = eigenvals_real[j+1, next_max_idx]
        
        if (curr_max_val < 0 and next_max_val > 0) or (curr_max_val > 0 and next_max_val < 0):
            for cond_idx in range(4):
                if stability_arr[j, cond_idx] * stability_arr[j+1, cond_idx] < 0:
                    interp = abs(curr_max_val) / (abs(curr_max_val) + abs(next_max_val))
                    x = eigenvals_real[j, curr_max_idx] * (1-interp) + eigenvals_real[j+1, next_max_idx] * interp
                    y = eigenvals_imag[j, curr_max_idx] * (1-interp) + eigenvals_imag[j+1, next_max_idx] * interp
                    lab = stability_labels[cond_idx] if not stability_used[cond_idx] else ""
                    stability_used[cond_idx] = True
                    ax.plot(x, y, color=stability_colors[cond_idx], marker='x', linestyle=None, markersize=10,
                            label=lab, markeredgewidth=1.5, zorder=4, alpha=0.7)
                    ax.plot(x, -y, color=stability_colors[cond_idx], marker='x', linestyle=None, markersize=10,
                            markeredgewidth=1.5, zorder=4, alpha=0.7) 
    
    ax.set_xlabel('Re(λ)')
    ax.set_ylabel('Im(λ)')
    init_perturb_str = f'[{init_perturb[0]:.1g}, {init_perturb[1]:.1g}, {init_perturb[2]:.1g}, {init_perturb[3]:.1g}]'
    init_digits = re.sub(r"\D", "", init_perturb_str)
    final_perturb_str = f'[{final_perturb[0]:.1g}, {final_perturb[1]:.1g}, {final_perturb[2]:.1g}, {final_perturb[3]:.1g}]'
    final_digits = re.sub(r"\D", "", final_perturb_str)
    fig.suptitle(f'eigenvalue trajectories, contrast={int(contrast * 100)}%, init={init_perturb_str}, final={final_perturb_str}',
                 y=0.98, fontsize=14)
    
    ax.grid(True, alpha=0.2)
    
    ax.legend(bbox_to_anchor=(1.02, 0.8), loc='upper left')
    plt.tight_layout()

    if savefig:
        os.makedirs("../figures", exist_ok=True)
        datet = datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = f"../figures/eigenvalue_scatter_k{k}_c{contrast}_start{init_digits}_end{final_digits}_{datet}.pdf"
        plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300, format='pdf')
        print(f"figure saved at {save_path}")

    return fig

def interp_zero_crossings(arr, x, y):
    crossings_x = []
    crossings_y = []
    for i in range(arr.shape[0] - 1):
        for j in range(arr.shape[1] - 1):
            if arr[i, j] * arr[i, j + 1] < 0:
                x_interp = x[j] + (x[j + 1] - x[j]) * abs(arr[i, j]) / (
                    abs(arr[i, j]) + abs(arr[i, j + 1]))
                crossings_x.append(x_interp)
                crossings_y.append(y[i])
            if arr[i, j] * arr[i + 1, j] < 0:
                y_interp = y[i] + (y[i + 1] - y[i]) * abs(arr[i, j]) / (
                    abs(arr[i, j]) + abs(arr[i + 1, j]))
                crossings_x.append(x[j])
                crossings_y.append(y_interp)
    return crossings_x, crossings_y
