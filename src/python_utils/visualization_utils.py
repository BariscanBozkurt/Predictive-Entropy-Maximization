import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Math, display

########### LATEX Style Display Matrix ###############
def display_matrix(array):
    """Display given numpy array with Latex format in Jupyter Notebook.

    Args:
        array (numpy array): Array to be displayed
    """
    data = ""
    for line in array:
        if len(line) == 1:
            data += " %.3f &" % line + r" \\\n"
            continue
        for element in line:
            data += " %.3f &" % element
        data += r" \\" + "\n"
    display(Math("\\begin{bmatrix} \n%s\\end{bmatrix}" % data))

def plot_bss_comparison(S, Y, title="Signal Comparison", figsize=(12, 10)):
    """
    S: Ground truth sources (n_sources, n_samples)
    Y: Estimated sources (n_sources, n_samples) - Must be pre-aligned/signed
    """
    n = S.shape[0]
    
    # # Normalize to Z-score (Mean=0, Std=1) for visual comparison
    # # This is essential because BSS has scale ambiguity
    # S_norm = (S - S.mean(axis=1, keepdims=True)) / (S.std(axis=1, keepdims=True) + 1e-9)
    # Y_norm = (Y - Y.mean(axis=1, keepdims=True)) / (Y.std(axis=1, keepdims=True) + 1e-9)
    
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1: axes = [axes]

    for i in range(n):
        # Plot signals
        axes[i].plot(S[i, :], label="Ground Truth", color="#1f77b4", alpha=0.7, linewidth=2)
        axes[i].plot(Y[i, :], label="Estimation", color="#ff7f0e", linestyle="--", linewidth=1.5)
        
        # Calculate local MSE for this channel
        mse = np.mean((S[i, :] - Y[i, :])**2)
        
        axes[i].set_ylabel(f"Source {i}")
        axes[i].legend(loc="upper right", frameon=True, fontsize='small')
        axes[i].grid(True, linestyle=':', alpha=0.6)
        axes[i].set_title(f"Channel {i} - Normalized MSE: {mse:.5f}", fontsize=10)

    plt.xlabel("Time Samples")
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
def subplot_1D_signals(
    X, title="", title_fontsize=20, figsize=(10, 5), linewidth=1, colorcode="#050C12"
):
    """Plot the 1D signals (each row from the given matrix)"""
    n = X.shape[0]  # Number of signals

    fig, ax = plt.subplots(n, 1, figsize=figsize)

    for i in range(n):
        ax[i].plot(X[i, :], linewidth=linewidth, color=colorcode)
        ax[i].grid()

    plt.suptitle(title, fontsize=title_fontsize)
    # plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
    plt.draw()

def visualize_flattened_images(I, imsize, title="", height=5, width=15, cmap='gray'):
    """
    Visualizes grayscale images from a matrix of flattened vectors.
    
    Args:
        I (ndarray): Matrix of shape (n_images, n_pixels).
        imsize (tuple): Original dimensions of the image (height, width).
        title (str): Global title for the figure.
        height (int): Figure height.
        width (int): Figure width.
        cmap (str): Colormap (default 'gray').
    """
    n_images = I.shape[0]
    
    # Create subplots in a single row
    fig, ax = plt.subplots(1, n_images, figsize=(width, height))
    fig.suptitle(title, fontsize=16)
    
    # Ensure ax is an array even for 1 image
    if n_images == 1:
        ax = [ax]
    
    for i in range(n_images):
        # Reshape the flattened vector back to 2D
        img_2d = I[i, :].reshape(imsize[0], imsize[1])
        
        # Display image
        # Using vmin/vmax with percentile clipping helps if there are BSS artifacts
        v_min, v_max = np.percentile(img_2d, [1, 99])
        ax[i].imshow(img_2d, cmap=cmap, vmin=v_min, vmax=v_max)
        
        # Cleanup axes
        ax[i].axis('off')
        ax[i].set_title(f"Channel {i}")

    plt.subplots_adjust(
        right=0.95, left=0.05, bottom=0.05, top=0.85, wspace=0.2
    )
    plt.show()
    
def create_summary_table(df):
    """
    Detects parameters and groups by Model + Params.
    Reports Mean ± 95% Confidence Interval.
    """
    # 1. Calculate Mean Component SNR
    df['mSNR'] = df['SNR'].apply(lambda x: np.mean(x[0]) if isinstance(x, list) else np.nan)

    # 2. Automatically determine grouping columns
    potential_params = ['rho', 'SNRinp']
    group_cols = ['Model'] + [col for col in potential_params if col in df.columns]

    # 3. Aggregate statistics: we need 'count' and 'std' to calculate SEM/CI
    summary = df.groupby(group_cols).agg({
        'SINR': ['mean', 'std', 'count'],
        'mSNR': ['mean', 'std', 'count'],
        'execution_time': ['mean', 'std', 'count']
    })

    def get_ci95(group_row, col_name):
        mean = group_row[(col_name, 'mean')]
        std = group_row[(col_name, 'std')]
        n = group_row[(col_name, 'count')]
        
        if n <= 1: return f"{mean:.2f} ± 0.00"
        
        # Calculate SEM and 95% CI Margin
        sem = std / np.sqrt(n)
        ci95_margin = stats.t.ppf(0.975, n - 1) * sem
        return f"{mean:.2f} ± {ci95_margin:.2f}"

    # 4. Format for Paper
    final = pd.DataFrame(index=summary.index)
    final['SINR (dB)'] = summary.apply(lambda x: get_ci95(x, 'SINR'), axis=1)
    final['mSNR (dB)'] = summary.apply(lambda x: get_ci95(x, 'mSNR'), axis=1)
    final['Time (s)'] = summary.apply(lambda x: get_ci95(x, 'execution_time'), axis=1)

    sort_levels = [col for col in potential_params if col in df.columns]
    if sort_levels:
        final = final.sort_index(level=sort_levels)

    return final

def plot_snr_performance(summary_df, x_axis_param='rho', title=None):
    """
    Revised generalized plotting function for publication-quality figures.
    - Accepts a custom title; otherwise uses defaults based on x_axis_param.
    - Renames models and ensures 'PredictiveDecor (Ours)' is first in legend.
    - Benchmarks are ranked by mean performance and rendered as dashed lines.
    - Specific simulation values are used as x-ticks.
    """
    # 1. Preprocess Dataframe: Rename Models to Publication Standards
    name_mapping = {
        'PredictiveDecorrBSS': 'PredictiveDecor (Ours)',
        'PredictiveBSS': 'PredictiveDecor (Ours)',
        'CorInfoMax': 'CorInfoMax',
        'LDMIBSS': 'LD-InfoMax',
        'LD-InfoMax': 'LD-InfoMax',
        'ICA_InfoMax': 'ICA-InfoMax',
        'ICA-InfoMax': 'ICA-InfoMax',
        'BSMBSS': 'BSM',
        'NSMBSS': 'NSM'
    }
    
    plot_data = summary_df.reset_index()
    plot_data['Model'] = plot_data['Model'].map(name_mapping).fillna(plot_data['Model'])
    
    # Parse "Mean ± CI_Margin"
    plot_data['mSNR_mean'] = plot_data['mSNR (dB)'].apply(lambda x: float(x.split(' ± ')[0]))
    plot_data['mSNR_ci'] = plot_data['mSNR (dB)'].apply(lambda x: float(x.split(' ± ')[1]))
    
    plt.figure(figsize=(13, 8), dpi=100)
    
    colors = {
        'PredictiveDecor (Ours)': '#d62728',
        'CorInfoMax': '#ff7f0e',
        'LD-InfoMax': '#7f7f7f',
        'ICA-InfoMax': '#1f77b4',
        'BSM': '#2ca02c',
        'NSM': '#9467bd'
    }

    model_ranking = plot_data.groupby('Model')['mSNR_mean'].mean().sort_values(ascending=False)
    sorted_benchmarks = [m for m in model_ranking.index if m != 'PredictiveDecor (Ours)']
    final_order = ['PredictiveDecor (Ours)'] + sorted_benchmarks

    x_ticks = sorted(plot_data[x_axis_param].unique())
    
    for model in final_order:
        if model not in plot_data['Model'].values: continue
            
        model_df = plot_data[plot_data['Model'] == model].sort_values(x_axis_param)
        x_vals = model_df[x_axis_param]
        mean = model_df['mSNR_mean']
        ci = model_df['mSNR_ci'] # This is now the 95% CI Margin
        
        is_ours = (model == 'PredictiveDecor (Ours)')
        plt.plot(x_vals, mean, label=model, color=colors.get(model, '#333333'), 
                 linestyle='-' if is_ours else '--', marker='o', 
                 linewidth=4.5 if is_ours else 2.5, markersize=10, zorder=10 if is_ours else 5)
        
        # Shaded area now represents the 95% CI
        plt.fill_between(x_vals, mean - ci, mean + ci, 
                         color=colors.get(model, '#333333'), alpha=0.15)

    # Formatting
    plt.ylabel('Mean Component SNR (mSNR) [dB] (95% CI)', fontsize=20)
    # ... (Rest of your plotting logic remains the same)
    
    # 5. Title Logic and Axis Formatting
    if title is not None:
        plt.title(title, fontsize=22, fontweight='bold', pad=30)
    elif x_axis_param == 'rho':
        plt.title('Performance Robustness to Source Correlation ($\\rho$)', fontsize=22, fontweight='bold', pad=30)
    elif x_axis_param == 'SNRinp':
        plt.title('Component Recovery vs. Noise Level', fontsize=22, fontweight='bold', pad=30)

    # Axis Labels and Scaling
    if x_axis_param == 'rho':
        plt.xlabel('Correlation Coefficient ($\\rho$)', fontsize=22)
    elif x_axis_param == 'SNRinp':
        plt.xlabel('Input SNR (dB)', fontsize=22)
        plt.gca().invert_xaxis() # Move from High SNR to Low SNR (Increasing Noise)
    
    plt.xticks(ticks=x_ticks, labels=x_ticks, fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=14, loc='best', frameon=True, shadow=True)
    plt.tight_layout()
    return plt