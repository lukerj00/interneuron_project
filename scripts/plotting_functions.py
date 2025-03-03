# imports
import numpy as np
import scipy
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import copy
import logging
import itertools
import os
from datetime import datetime
import re

from .fitting_functions import get_ISN_coeffs, get_fit_poly_1, get_fit_poly_2, fixed_point_refinement, get_fp_condition

def plot_data_EPSV(c_range, means, sems, savefig=False):
    fs = 24
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    d, n = means.shape
    ms = 32
    lw = 0.8
    cw = 1.4
    colors = ['#35322F', '#3FA4BC', '#EA9523', '#F2948F']
    labels = ['$r_{E}$', '$r_{P}$', '$r_{S}$', '$r_{V}$']
    
    for i in range(d):
        plt.plot(c_range, means[i], color=colors[i], label=labels[i])
        
        plt.scatter(c_range, means[i], edgecolor=colors[i], facecolors='none', 
                   s=ms, marker='o', linewidth=cw)
        
        plt.errorbar(c_range, means[i], yerr=sems[i], color=colors[i],
                    fmt='none', capsize=5, capthick=cw, elinewidth=lw)

    plt.xlabel('contrast', fontsize=fs)
    plt.ylabel('event rate/mean', fontsize=fs)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_linewidth(lw)
    plt.gca().spines['left'].set_linewidth(lw)
    
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.xticks([0, 0.5, 1])
    plt.gca().set_xticklabels(['0', '0.5', '1'])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.gca().set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    plt.locator_params(axis='y', nbins=6)
    plt.tick_params(axis='both', length=2, width=lw, direction='in')
    plt.margins(x=0.025)
    plt.ylim(0, 1)
    leg = plt.legend(fontsize=fs, loc='upper left', frameon=False, bbox_to_anchor=(1.02, 0.75))

    for legobj in leg.get_lines():
        legobj.set_linewidth(2.0)
    
    plt.show()

    if savefig:
        os.makedirs("../figures", exist_ok=True)
        datet = datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = f"../figures/V1_data_{datet}.pdf"
        plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300, format='pdf')
        print(f"figure saved at {save_path}")

    return fig

def plot_data_fits_EPSV(c_range, means, top_fits, n_plot=10, savefig=False):
    fs = 24  
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    d, n = means.shape 
    ms = 32 
    lw = 0.8 
    cw = 1.4 
    alpha = 0.5 
    colors = ['#35322F', '#3FA4BC', '#EA9523', '#F2948F']
    labels = ['$r_{E}$', '$r_{P}$', '$r_{S}$', '$r_{V}$']
    
    for sample in top_fits[:n_plot]:
        for i in range(d):
            plt.plot(c_range, sample[i], alpha=alpha, color=colors[i], linestyle='--')

    for i in range(d):
        plt.plot(c_range, means[i], color=colors[i], label=labels[i])
        plt.scatter(c_range, means[i], edgecolor=colors[i], facecolors='none',
                    s=ms, marker='o', linewidth=cw)

    plt.xlabel('contrast', fontsize=fs)
    plt.ylabel('$r_{X}$', fontsize=fs)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_linewidth(lw)
    plt.gca().spines['left'].set_linewidth(lw)
    
    plt.tick_params(axis='both', which='major', labelsize=fs)
    plt.xticks([0, 0.5, 1])
    plt.gca().set_xticklabels(['0', '0.5', '1'])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.gca().set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    plt.locator_params(axis='y', nbins=6)
    plt.tick_params(axis='both', length=2, width=lw, direction='in')
    plt.margins(x=0.025)
    plt.ylim(0, 1)
    
    leg = plt.legend(fontsize=fs, loc='upper left', frameon=False, bbox_to_anchor=(1.02, 0.75))
    for legobj in leg.get_lines():
        legobj.set_linewidth(2.0)

    plt.show()

    if savefig:
        os.makedirs("../figures", exist_ok=True)
        datet = datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = f"../figures/V1_data_and_top{n_plot}_fits_{datet}.pdf"
        plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300, format='pdf')
        print(f"figure saved at {save_path}")

    return fig

def plot_W_heatmap(W, k, lim=3, savefig=False):
    W_norm = np.log((np.abs(W) + 1e-10)/(np.abs(W[0, 1]) + 1e-10)) # consts for stability
    max_value = np.ceil(np.max(W_norm))

    fig, ax = plt.subplots(figsize=(6, 6), facecolor='none')
    cmap = plt.cm.Greys.copy()
    cmap.set_under('white')
    norm = matplotlib.colors.Normalize(vmin=-lim, vmax=max_value, clip=False)
    im = ax.imshow(W_norm, cmap=cmap, interpolation='nearest', norm=norm)
    
    labels = ['E', 'PV', 'SOM', 'VIP']
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=18)
    ax.set_yticklabels(labels, fontsize=18)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('pre-synaptic', fontsize=20, labelpad=20)
    ax.set_ylabel('post-synaptic', fontsize=20, labelpad=10)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_aspect('equal')

    cax = fig.add_axes([0.96, 0.08, 0.05, 0.82]) # [left, bottom, width, height]
    
    cbar = plt.colorbar(im, cax=cax, orientation='vertical', pad=0.02)
    cbar.set_ticks([-lim, 0, max_value])
    cbar.set_ticklabels([f'{int(-lim)}', '0', f'{int(max_value)}'])
    cbar.ax.tick_params(labelsize=18, length=0)
    cbar.ax.set_title(r'$\log \left( \frac{ \langle w \rangle}{\langle w_{EP} \rangle} \right) $', fontsize=20, pad=28)
    
    if savefig:
        os.makedirs("../figures", exist_ok=True)
        datet = datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = f"../figures/W_heatmap_k{k}_{datet}.pdf"
        plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300, format='pdf')
        print(f"figure saved at {save_path}")

    return fig

def plot_W_hists_all(top_params, bins, lower_lim=-3, percentiles=[1,99], savefig=False):
    top_W = [p[0] for p in top_params]
    pop_labels = ['E', 'P', 'S', 'V']
    fsize = 28
    n_params = len(top_W)
    d = top_W[0].shape[0]

    colors = {'E': 'black', 'PV': '#3FA4BC', 'SOM': '#EA9523', 'VIP': '#F2948F'}
    labels = ['E', 'PV', 'SOM', 'VIP']

    W_avg = np.mean([W for W in top_W], axis=0)
    W_avg_norm = np.log((np.abs(W_avg) + 1e-10)/(np.abs(W_avg[0, 1]) + 1e-10))  # constz for stability
    W_avg_norm[W_avg == 0] = -np.inf
    max_value = np.ceil(np.max(W_avg_norm) + 0.5)

    fig, axs = plt.subplots(4, 4, figsize=(10, 10))
    cmap = plt.cm.Greys
    cmap.set_under('white')
    norm = matplotlib.colors.Normalize(vmin=lower_lim, vmax=max_value, clip=False)

    column_ranges = []
    for j in range(d): 
        all_column_values = []
        for W in top_W:
            all_column_values.extend(W[:, j])
        all_column_values = [v for v in all_column_values if v != -np.inf]

        p_low = np.percentile(all_column_values, percentiles[0])  # th percentile
        p_high = np.percentile(all_column_values, percentiles[1])  # nth percentile
        column_ranges.append((p_low, p_high))

    for i in range(d):
        for j in range(d):
            ax = axs[i, j]

            background_value = W_avg_norm[i, j]
            ax.set_facecolor(cmap(norm(background_value)))

            values = [W[i, j] for W in top_W]
            values = [v for v in values if v != -np.inf]

            if values:
                p_low, p_high = column_ranges[j]
                filtered_values = [v for v in values if p_low <= v <= p_high]

                ax.hist(filtered_values, bins=bins, alpha=0.7,
                        color=colors[labels[i]],
                        range=(p_low, p_high))  

                mean_val = np.mean(filtered_values)
                background_intensity = (background_value - lower_lim) / (max_value - lower_lim)
                line_color = 'lightgrey' if background_intensity > 0.4 else '#555555'
                ax.axvline(mean_val, color=line_color, linestyle=':', alpha=0.8, linewidth=1.6)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect(1.0 / ax.get_data_ratio())

            delta = 0.03
            if i == 3:
                xmin, xmax = ax.get_xlim()
                ax.text(xmin, ax.get_ylim()[0] - delta * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                        f'{p_low:.0f}', ha='left', va='top', fontsize=20) # column_ranges[j][0]:.0f
                ax.text(xmax, ax.get_ylim()[0] - delta * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                        f'{p_high:.0f}', ha='right', va='top', fontsize=20)

            if j == 0:
                label_color = colors[labels[i]]
                ax.set_ylabel(labels[i], fontsize=fsize, labelpad=15, weight='bold', color=label_color)
            if i == 0:
                label_color = colors[labels[j]]
                ax.set_title(labels[j], fontsize=fsize, weight='bold', color=label_color)

            ax.text(0.5, -0.1, f'$w_{{{pop_labels[i]}{pop_labels[j]}}}$',
                    transform=ax.transAxes,
                    ha='center',
                    va='center',
                    fontsize=22)

    fig.text(0.52, 1.01, 'pre-synaptic', ha='center', va='top', fontsize=28, weight='bold')
    fig.text(0.04, 0.54, 'post-synaptic', va='center', rotation='vertical', fontsize=28, weight='bold')

    cax = fig.add_axes([0.17, 0.06, 0.66, 0.03])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=cax, orientation='horizontal')

    cb.set_ticks([lower_lim, 0, max_value])
    cb.set_ticklabels([f'{int(lower_lim)}', '0', f'{int(max_value)}'])
    cb.ax.tick_params(labelsize=22, length=0)

    fig.text(0.86, 0.06, r'$\log \left( \frac{ \langle w \rangle}{\langle w_{EP} \rangle} \right) $',
             fontsize=24)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.15, hspace=0.25, wspace=-0.55)
    plt.show()

    if savefig:
        os.makedirs("../figures", exist_ok=True)
        datet = datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = f"../figures/W_hists_top{n_params}_fits_{datet}.pdf"
        plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300, format='pdf')
        print(f"figure saved at {save_path}")

    return fig

def plot_top_likelihoods(log_likelihoods, top_k_label=10, bins=50, proportion=1.0, savefig=False):
    N = len(log_likelihoods)
    log_likelihoods = np.array(log_likelihoods)
    sorted_log_likelihoods = np.sort(log_likelihoods)[::-1]
    
    n_points = int(len(sorted_log_likelihoods) * proportion)
    sorted_log_likelihoods = sorted_log_likelihoods[:n_points]
    
    if top_k_label > n_points:
        raise ValueError(f"top_k ({top_k_label}) exceeds the number of points considered ({n_points}). Reduce `top_k` or increase `prop`.")
    
    bin_edges = np.linspace(0, n_points, bins + 1, dtype=int)
    binned_means = [np.mean(sorted_log_likelihoods[bin_edges[i]:bin_edges[i + 1]]) for i in range(len(bin_edges) - 1)]
    binned_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
    binned_widths = np.diff(bin_edges)
        
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ymin = 0 #1 * np.min(sorted_log_likelihoods)
    ymax = 1.02 * np.max(sorted_log_likelihoods)
    ax.set_ylim([ymin, ymax])

    ax.bar(binned_centers, binned_means, width=binned_widths, align='center',
           color='#D3D3D3', edgecolor='black', linewidth=1.2)
    ax.axvline(x=top_k_label, color='red', linestyle='--', linewidth=1.5, label=f'top_k: Top {top_k_label}', ymin=ymin, ymax=ymax)
    ax.text(top_k_label, ax.get_ylim()[1], f'n={top_k_label}', 
            horizontalalignment='center', verticalalignment='bottom', fontsize=14, color='red')
    
    ax.set_xlabel('no. fits', fontsize=14)
    ax.set_ylabel('Log Likelihood', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
        
    plt.tight_layout()
    plt.show()

    if savefig:
        os.makedirs("../figures", exist_ok=True)
        datet = datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = f"../figures/likelihoods_hist_N{N}_{datet}.pdf"
        plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300, format='pdf')
        print(f"figure saved at {save_path}")
    
    return fig

def plot_ISN_hists(top_params, c_range, top_fits, bins=10, k=99, savefig=False):
    N = len(top_params)
    fs = 16

    all_isn_coeffs = [get_ISN_coeffs(W, h1, h0, c_range, fit) for (W,h1,h0), fit in zip(top_params, top_fits)]
    all_isn_coeffs = np.array(all_isn_coeffs).T 

    coeff_range = [np.min(all_isn_coeffs), np.percentile(all_isn_coeffs, k)]
    global_bins = np.linspace(coeff_range[0], coeff_range[1], bins)
    max_freq = np.max([np.histogram(coeffs, bins=global_bins)[0] for coeffs in all_isn_coeffs])
    N_c = all_isn_coeffs.shape[0] 
    fig, axes = plt.subplots(1, N_c, figsize=(4 * N_c, 4), sharey=True)

    for i in range(N_c):
        coeffs = all_isn_coeffs[i]
        axes[i].hist(coeffs, bins=global_bins, color='black', alpha=0.7, density=True)
        axes[i].set_xlim([-1, np.ceil(coeff_range[1])])
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['bottom'].set_linewidth(1)
        axes[i].spines['left'].set_linewidth(1)
        axes[i].set_xlabel(f'contrast = {c_range[i]*100}%', fontsize=fs)
        axes[i].tick_params(axis='both', which='major', labelsize=fs, top=False, right=False)
        axes[i].set_xticks(np.linspace(-1, np.ceil(coeff_range[1]), int(np.ceil(coeff_range[1]) + 2), dtype=int))
        if i == 0:
            axes[i].set_yticks(np.linspace(0, np.ceil(max_freq), int(np.ceil(max_freq)) + 1, dtype=int))
            axes[i].set_ylabel('Frequency', fontsize=fs)
        
    plt.tight_layout()
    plt.show()

    if savefig:
        os.makedirs("../figures", exist_ok=True)
        datet = datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = f"../figures/ISN_hists_N{N}_{datet}.pdf"
        plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300, format='pdf')
        print(f"figure saved at {save_path}")

    return fig

def plot_polynomial(params, c_range, means, contrast, h_ext, TAU, k, 
                    z_lims=[-1, 10], y_lims=[-10, 10], N_z=100000, fit_condition=1, ax=None, 
                    alpha=1.0, lw=1.0, color='black', added_labels=None, label=None, savefig=False):
    fs = 22
    W, h1, h0 = params

    main_fp_refined = []
    fp_conditions = []
    if fit_condition == 1:
        main_fp, fixed_points, fp_errors, fp_labels, z, F_z, z_root_list, cond_a, cond_b = get_fit_poly_1(
            W, h1, h0, contrast, c_range, means, z_lims=z_lims, N_z=N_z
        )
        z_ind = 0
    elif fit_condition == 2:
        main_fp, fixed_points, fp_errors, fp_labels, z, F_z, z_root_list, cond_a, cond_b = get_fit_poly_2(
            W, h1, h0, contrast, c_range, means, z_lims=z_lims, N_z=N_z
        )
        z_ind = 1
    else:
        raise ValueError("invalid fit condition - choose 1 or 2")

    for fp in fixed_points:
        fp_refined = fixed_point_refinement(fp, W, h1, h0, contrast)
        main_fp_refined.append(fp_refined)
        fp_condition, J = get_fp_condition(fp_refined, W, h1, h0, contrast, TAU)
        fp_conditions.append(fp_condition)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = ax.get_figure()

    ax.axhline(0, color='red', linestyle='--', zorder=0, alpha=0.5)
    ax.plot(z, F_z, color=color, linewidth=lw, zorder=1, alpha=alpha, label=label)

    if added_labels is None:
        added_labels = set()

    for i in range(len(fixed_points)):
        stability_label = None
        if fp_conditions[i] == 0 and 'stable FP' not in added_labels:  # Stable
            stability_label = 'stable FP'
            added_labels.add('stable FP')
        elif fp_conditions[i] == 1 and 'unstable FP' not in added_labels:  # Unstable
            stability_label = 'unstable FP'
            added_labels.add('unstable FP')
        elif fp_conditions[i] == 2 and 'saddle FP' not in added_labels:  # Saddle
            stability_label = 'saddle FP'
            added_labels.add('saddle FP')

        stability_color = 'black' if fp_conditions[i] == 0 else 'red' if fp_conditions[i] == 1 else 'blue'
        ax.scatter(fixed_points[i][z_ind]**0.5, 0, color=stability_color, s=20, zorder=2, label=stability_label)

    ax.set_xlabel(r'$z$', fontsize=fs)
    ax.set_ylabel(r'$\mathcal{F}(z)$', fontsize=fs)
    ax.set_xlim([z_lims[0], z_lims[1] + 1])
    ax.set_ylim(y_lims)
    
    x_step = 10 if z_lims[1] >= 20 else 5
    ax.set_xticks(np.arange(z_lims[0], z_lims[1] + 1, x_step))
    ax.set_yticks(np.arange(y_lims[0], y_lims[1] + 1, 5))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='both', which='major', length=3, width=1.5, labelsize=fs)

    if added_labels or label:
        ax.legend(loc='upper right', fontsize=fs - 4, frameon=False)

    if savefig:
        os.makedirs("../figures", exist_ok=True)
        datet = datetime.now().strftime("%y%m%d_%H%M%S")
        h_ext_str = re.sub(r"\D", "", str(h_ext))
        save_path = f"../figures/poly_fit_k{k}_c{contrast}_hext{h_ext_str}_fc{fit_condition}_{datet}.pdf"
        plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300, format='pdf')
        print(f"figure saved at {save_path}")    

    return fig, ax, added_labels

def plot_polynomial_c_range(params, c_range, means, h_ext, TAU, k, 
                            z_lims=[-1, 10], y_lims=[-10, 10], N_z=100000, fit_condition=1, savefig=False):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    added_labels = set()
    blues = ['#BFD7ED', '#9EC5E8', '#7EB3E3', '#4193D4', '#217FCE', '#0B6BC8']
    
    for i, contrast in enumerate(c_range):
        fig, ax, added_labels = plot_polynomial(
            params, c_range, means, contrast, h_ext, TAU, k, 
            z_lims=z_lims, y_lims=y_lims, N_z=N_z, fit_condition=fit_condition, ax=ax, 
            alpha=0.8, lw=1.4, color=blues[i], added_labels=added_labels, label=f'contrast {int(100*contrast)}%' 
        )
    
    z_range = z_lims[1] - z_lims[0]
    y_range = y_lims[1] - y_lims[0]
    
    if z_range <= 5:
        ax.xaxis.set_major_locator(plt.MultipleLocator(1)) 
    elif z_range <= 10:
        ax.xaxis.set_major_locator(plt.MultipleLocator(2)) 
    else:
        ax.xaxis.set_major_locator(plt.MaxNLocator(5)) 
    
    if y_range <= 5:
        ax.yaxis.set_major_locator(plt.MultipleLocator(1)) 
    elif y_range <= 20:
        ax.yaxis.set_major_locator(plt.MultipleLocator(5)) 
    else:
        ax.yaxis.set_major_locator(plt.MaxNLocator(5)) 
    
    handles, labels = ax.get_legend_handles_labels()
    contrast_indices = [i for i, label in enumerate(labels) if 'contrast' in label]
    stability_indices = [i for i, label in enumerate(labels) if 'contrast' not in label]
    ordered_handles = [handles[i] for i in contrast_indices] + [handles[i] for i in stability_indices]
    ordered_labels = [labels[i] for i in contrast_indices] + [labels[i] for i in stability_indices]
    
    ax.legend(ordered_handles, ordered_labels, fontsize=16, loc='upper right', frameon=False, bbox_to_anchor=(1.25, 1.1))
    ax.tick_params(axis='both', which='major', length=5, width=1.5, labelsize=14)
    ax.tick_params(axis='both', which='minor', length=3, width=1.0)
    ax.margins(x=0.05, y=0.05)
    
    if savefig:
        os.makedirs("../figures", exist_ok=True)
        datet = datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = f"../figures/poly_fit_k{k}_all_contrasts_fc{fit_condition}_{datet}.pdf"
        plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300, format='pdf')
        print(f"figure saved at {save_path}")
    
    return fig

def plot_polynomial_multiple_fits(indices, params_list, c_range, means, h_ext, contrast, TAU, 
                                z_lims=[-1, 10], y_lims=[-10, 10], N_z=100000, fit_condition=1, savefig=False):
    n_params = len(params_list)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    added_labels = set()
    colors = ['#4285F4', '#EA4335', '#FBBC05', '#34A853', '#8B44AC', '#0F9D58', '#DB4437', '#4285F4']
    
    if len(indices) > len(colors):
        colors = plt.cm.tab10(np.linspace(0, 1, len(indices)))
    
    for i in range(n_params):
        params = params_list[i]
        color = colors[i] if i < len(colors) else colors[i % len(colors)]
        
        fig, ax, added_labels = plot_polynomial(
            params, c_range, means, contrast, h_ext, TAU, i, 
            z_lims=z_lims, y_lims=y_lims, N_z=N_z, 
            fit_condition=fit_condition, ax=ax, 
            alpha=0.8, lw=1.4, color=color, 
            added_labels=added_labels,
            label=None
        )
    
    z_range = z_lims[1] - z_lims[0]
    y_range = y_lims[1] - y_lims[0]
    
    if z_range <= 5:
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    elif z_range <= 10:
        ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    else:
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    
    if y_range <= 5:
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    elif y_range <= 20:
        ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    else:
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    
    handles, labels = ax.get_legend_handles_labels()
    fit_indices = [i for i, label in enumerate(labels) if 'fit' in label]
    stability_indices = [i for i, label in enumerate(labels) if 'fit' not in label]
    ordered_handles = [handles[i] for i in fit_indices] + [handles[i] for i in stability_indices]
    ordered_labels = [labels[i] for i in fit_indices] + [labels[i] for i in stability_indices]
    
    ax.legend(ordered_handles, ordered_labels, fontsize=16, loc='upper right', frameon=False, bbox_to_anchor=(1.02, 1.1))
    ax.tick_params(axis='both', which='major', length=5, width=1.5, labelsize=14)
    
    if savefig:
        os.makedirs("../figures", exist_ok=True)
        datet = datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = f"../figures/poly_fit_top{n_params}_c{contrast}_fc{fit_condition}_{datet}.pdf"
        plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300, format='pdf')
        print(f"figure saved at {save_path}")
    
    return fig

def plot_trajectory(x_plot, r_fp, W, h_ext, h_tot, variable, contrast, k, T_sim=10, dt=1e-4, sim_type='', noise_type='no', savefig=False):
    tot_steps = int(T_sim/dt)
    steps = np.arange(tot_steps)
    fs = 24 
    lw = 0.8 
    plw = 1.5 
    alpha_fp = 0.3 
    alpha_plot = 0.8
    alpha_n = 0.5 
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if variable == 'rate':
        var = 'r'
        x_dotted = r_fp
    elif variable == 'voltage':
        var = 'v'
        x_dotted = W @ r_fp + h_tot
    else:
        raise ValueError("invalid variable, choose 'rate' or 'voltage'")
    
    y_max = max(np.nanmax(x_plot), np.max(x_dotted) if len(x_dotted) > 0 else -np.inf)
    y_min = min(np.nanmin(x_plot), np.min(x_dotted) if len(x_dotted) > 0 else np.inf)
    y_range = y_max - y_min
    padding = 0.05 * y_range
    y_max_padded = y_max + padding
    y_min_padded = y_min - padding
    
    if not np.all(np.isfinite(x_plot)) or y_max > 20:
        if not np.isfinite(y_max):
            y_max = 1e6
        safe_ticks = [0, y_max]
        safe_labels = ["0", f"{y_max:.1e}"]
        plt.yticks(safe_ticks, safe_labels)
    else:
        ax.set_ylim(y_min_padded, y_max_padded)
        
        locator = MaxNLocator(nbins=5, steps=[1, 2, 5, 10], integer=False, min_n_ticks=3)
        ax.yaxis.set_major_locator(locator)
        def tick_formatter(x, pos):
            if abs(x) < 1e-10: 
                return "0"
            elif abs(x) >= 1e3 or abs(x) <= 1e-2:
                return f"{x:.2e}" 
            elif abs(x - round(x)) < 1e-10:
                return f"{int(x)}"
            else:
                formatted = f"{x:.2f}"
                if formatted.endswith('0'):
                    return formatted[:-1]
                else:
                    return formatted
        ax.yaxis.set_major_formatter(plt.FuncFormatter(tick_formatter))
    
    var_colors = ['#35322F', '#3FA4BC', '#EA9523', '#F2948F']
    noise_color = 'grey'
    
    var_labels = [f"${{{var}}}_{{E}}$", f"${{{var}}}_{{P}}$", f"${{{var}}}_{{S}}$", f"${{{var}}}_{{V}}$"]
    noise_label = f"$\eta_{{EPSV}}$"
    
    for color, val in zip(var_colors, x_dotted):
        plt.axhline(y=val, color=color, linestyle='--', alpha=alpha_fp, linewidth=plw, zorder=0)
    
    for i, (color, label) in enumerate(zip(var_colors, var_labels)):
        data = x_plot[:, i]
        masked_data = np.ma.masked_invalid(data)
        plt.plot(steps, masked_data, color=color, label=label, alpha=alpha_plot, linewidth=plw, zorder=2)
        
        if np.any(np.isnan(data)):
            first_nan_idx = np.where(np.isnan(data))[0][0]
            
            if first_nan_idx > 0:
                last_valid_idx = first_nan_idx - 1
                last_valid_y = data[last_valid_idx]
                plt.plot(steps[last_valid_idx], last_valid_y, 'o', color=color, markersize=8, zorder=3)
    
    for i in range(x_plot.shape[1] - len(var_labels)):
        idx = i + len(var_labels)
        data = x_plot[:, idx]
        masked_data = np.ma.masked_invalid(data)
        
        if i == 0:
            plt.plot(steps, masked_data, color=noise_color, label=noise_label,
                     linewidth=plw, alpha=alpha_n, zorder=1)
        else:
            plt.plot(steps, masked_data, color=noise_color,
                     linewidth=plw, alpha=alpha_n, zorder=1)
    
    ax.set_xlim(0, tot_steps)    
    plt.xlabel('$t$', fontsize=fs)
    plt.ylabel(f"${var}(t)$", fontsize=fs)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(lw)
    ax.spines['left'].set_linewidth(lw)
    plt.tick_params(axis='both', which='major', labelsize=fs)
    plt.tick_params(axis='both', length=2, width=lw, direction='in')
    
    t_max = len(steps)
    x_ticks = [0, t_max // 2, t_max]
    x_tick_labels = [f"{int(round(dt * x))}" for x in x_ticks]
    plt.xticks(x_ticks, x_tick_labels)

    leg = plt.legend(fontsize=fs, loc='upper left', frameon=False, bbox_to_anchor=(1.02, 0.8))
    for legobj in leg.get_lines():
        legobj.set_linewidth(2.0)
    
    plt.margins(x=0.025)
    plt.tight_layout()
    plt.show()
    
    if savefig:
        os.makedirs("../figures", exist_ok=True)
        datet = datetime.now().strftime("%y%m%d_%H%M%S")
        h_ext_str = re.sub(r"\D", "", str(h_ext))
        save_path = f"../figures/sim_{sim_type}_k{k}_c{contrast}_hext{h_ext_str}_{variable}_{noise_type}_noise_{datet}.pdf"
        plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300, format='pdf')
        print(f"figure saved at {save_path}")
    
    return fig

def plot_unique_correlations(online_covs, lyapunov_cov, variable, noise_type, normalised=False, savefig=False):
    N, d, _ = online_covs.shape
    var_labels = ['E', 'P', 'S', 'V']
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    if normalised:
        online_matrix = np.zeros_like(online_covs)
        for t in range(N):
            std_devs = np.sqrt(np.diagonal(online_covs[t]))
            online_matrix[t] = online_covs[t] / np.outer(std_devs, std_devs)
        
        std_devs_lyap = np.sqrt(np.diagonal(lyapunov_cov))
        lyapunov_matrix = lyapunov_cov / np.outer(std_devs_lyap, std_devs_lyap)
        
        y_label = 'corr estimate'
        title_prefix = 'online correlation'
        subscript_prefix = '\\rho'
        file_prefix = 'online_corr'
    else:
        online_matrix = online_covs
        lyapunov_matrix = lyapunov_cov
        
        y_label = 'cov estimate'
        title_prefix = 'online covariance'
        subscript_prefix = '\\sigma^2'
        file_prefix = 'online_cov'
    
    for i in range(d):
        for j in range(i, d):  # upper triangle
            values = online_matrix[:, i, j]
            subscript_str = f'{var_labels[i]}{var_labels[j]}'
            label = f"${subscript_prefix}_{{{subscript_str}}}$"
            
            line, = plt.plot(range(1, N + 1), lyapunov_matrix[i, j] * np.ones(N), alpha=0.5, linestyle='--')
            color = line.get_color()
            
            plt.plot(range(1, N + 1), values, label=label, color=color)
    
    plt.xlabel('timesteps', fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(f'{title_prefix} estimate, {variable} + {noise_type}', fontsize=14, y=1.02)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.tick_params(axis='both', length=3, width=1.5)
    
    ncols = min(5, d*(d+1)//2) 
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1.32), ncols=ncols)
    
    plt.show()
    
    if savefig:
        os.makedirs("../figures", exist_ok=True)
        datet = datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = f"../figures/{file_prefix}_estimate_{variable}_{noise_type}_noise_{datet}.pdf"
        plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300, format='pdf')
        print(f"figure saved at {save_path}")
    
    return fig

def plot_covariance_errors(online_covs, cov_errors, variable, noise_type, savefig=False):
    N, d, _ = online_covs.shape
    var_labels = ['E', 'P', 'S', 'V']
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    total_errors = np.sum(np.abs(cov_errors), axis=(1, 2))
    plt.plot(range(1, N + 1), total_errors, color='black', linewidth=1.5, label='Total Error')
    
    for i in range(d):
        for j in range(i, d):  # upper triangle
            error_values = np.abs(cov_errors[:, i, j])
            subscript_str = f'{var_labels[i]}{var_labels[j]}'
            label = f"$\\epsilon_{{{subscript_str}}}$"
            
            line, = plt.plot(range(1, N + 1), np.zeros(N), alpha=0.5, linestyle='--')
            color = line.get_color()    
            
            plt.plot(range(1, N + 1), error_values, label=label, color=color)
    
    plt.xlabel('timesteps', fontsize=14)
    plt.ylabel('cov estimation error', fontsize=14)
    plt.title(f'covariance estimation error, {variable} + {noise_type}', fontsize=14, y=1.02)
    plt.yscale('log')
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.locator_params(axis='x', nbins=5)
    plt.tick_params(axis='both', length=3, width=1.5)
    
    ncols = min(5, d*(d+1)//2 + 1) + 1 # + 1 for total error
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1.32), ncols=ncols)
    plt.show()
    
    if savefig:
        os.makedirs("../figures", exist_ok=True)
        datet = datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = f"../figures/cov_errors_timeseries_{variable}_{noise_type}_noise_{datet}.pdf"
        plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300, format='pdf')
        print(f"figure saved at {save_path}")
    
    return fig

def plot_cov_grid(variances_sim, variances_lyap, noise_lims, n_levels, contrast, variable, noise_type, savefig=False):
    var_labels = ['E', 'P', 'S', 'V']
    noise_levels = np.linspace(noise_lims[0], noise_lims[1], n_levels)
    N_a = len(var_labels)
    fig, axes = plt.subplots(N_a, N_a, figsize=(15, 15))
    fig.suptitle(f'output cov distribution, {variable} + {noise_type}, contrast = {contrast*100}%', fontsize=16)
    
    tick_positions = [noise_levels[0], 
                     noise_levels[len(noise_levels)//2],
                     noise_levels[-1]]
    
    def to_1sf(x):
        if x == 0:
            return '0'
        return f'{x:.1e}'.replace('e+0', 'e').replace('e-0', 'e-')
    
    row_ylims = []
    for i in range(N_a):
        row_min = float('inf')
        row_max = float('-inf')
        for j in range(N_a):
            sim_vars = variances_sim[:, i, j, :]
            lyap_vars = variances_lyap[:, i, j, :]
            
            mean_sim = np.mean(sim_vars, axis=0)
            sem_sim = np.std(sim_vars, axis=0) / np.sqrt(sim_vars.shape[0])
            mean_lyap = np.mean(lyap_vars, axis=0)
            sem_lyap = np.std(lyap_vars, axis=0) / np.sqrt(lyap_vars.shape[0])
            
            row_min = min(row_min, 
                         np.min(mean_sim - sem_sim),
                         np.min(mean_lyap - sem_lyap))
            row_max = max(row_max, 
                         np.max(mean_sim + sem_sim),
                         np.max(mean_lyap + sem_lyap))
        
        padding = (row_max - row_min) * 0.1
        row_ylims.append((row_min - padding, row_max + padding))
    
    for i in range(N_a):
        for j in range(N_a):
            ax = axes[i, j]
            
            sim_vars = variances_sim[:, i, j, :]
            lyap_vars = variances_lyap[:, i, j, :]
            
            mean_sim = np.mean(sim_vars, axis=0)
            sem_sim = np.std(sim_vars, axis=0) / np.sqrt(sim_vars.shape[0])
            ax.plot(noise_levels, mean_sim, '-', color='blue', label='Simulation')
            ax.fill_between(noise_levels, mean_sim-sem_sim, mean_sim+sem_sim,
                          color='blue', alpha=0.2)
            
            mean_lyap = np.mean(lyap_vars, axis=0)
            sem_lyap = np.std(lyap_vars, axis=0) / np.sqrt(lyap_vars.shape[0])
            ax.plot(noise_levels, mean_lyap, '--', color='red', label='Lyapunov')
            ax.fill_between(noise_levels, mean_lyap-sem_lyap, mean_lyap+sem_lyap,
                          color='red', alpha=0.2)
            
            ax.set_ylim(row_ylims[i])
            ax.set_xticks(tick_positions)
            y_min, y_max = row_ylims[i]
            y_range = np.linspace(y_min, y_max, 3)
            ax.set_yticks(y_range)
            ax.set_yticklabels([to_1sf(y) for y in y_range])
            
            if i != N_a-1:
                ax.set_xticks([])
            if j != 0:
                ax.set_yticks([])
            
            if i == N_a-1:
                ax.set_xlabel(f'$\sigma_{var_labels[j]}$')
            if j == 0:
                ax.set_ylabel(f'$\Sigma_{var_labels[i]}$')
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if i != N_a-1:
                ax.spines['bottom'].set_visible(False)
            if j != 0:
                ax.spines['left'].set_visible(False)
            if i == 0 and j == 0:
                ax.legend()

            if i == N_a-1 and j == N_a//2:
                fig.text(0.5, -0.01, 'Input variance', ha='center', va='center', fontsize=16)
            if i == N_a//2 and j == 0:
                fig.text(-0.01, 0.5, 'Output variance', ha='center', va='center', rotation='vertical', fontsize=16)

    plt.tight_layout()
    plt.show()

    if savefig:
        os.makedirs("../figures", exist_ok=True)
        datet = datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = f"../figures/cov_grid_{variable}_{noise_type}_noise_{datet}.pdf"
        plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=300, format='pdf')
        print(f"figure saved at {save_path}")

    return fig