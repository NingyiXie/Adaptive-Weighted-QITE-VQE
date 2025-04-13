import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np


def create_dual_axis_plots(data, method_circuits, metrics, 
                          x_labels=None, left_y_labels=None, right_y_labels=None, 
                          titles=None, colors=None, figsize=(18, 6)):
    """
    Create dual y-axis plots for quantum algorithm comparison with specific formatting.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the data with columns 'method_circuit', 'metric', 'iter', 'value'
    method_circuits : list
        List of method_circuit names to plot
    metrics : list
        List of metrics to plot (should be length 2)
    x_labels : list, optional
        Custom x-axis labels for each subplot
    left_y_labels : list, optional
        Custom left y-axis labels for each subplot
    right_y_labels : list, optional
        Custom right y-axis labels for each subplot
    titles : list, optional
        Custom titles for each subplot
    colors : dict, optional
        Custom colors for each metric
    figsize : tuple, optional
        Figure size (width, height)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    if len(metrics) != 2:
        raise ValueError("Exactly two metrics required for dual y-axis plots")
    
    # Set default values if not provided
    if x_labels is None:
        x_labels = ['Iterations'] * len(method_circuits)
    if left_y_labels is None:
        left_y_labels = [metrics[0]] * len(method_circuits)
    if right_y_labels is None:
        right_y_labels = [metrics[1]] * len(method_circuits)
    if titles is None:
        titles = [f"{method} Algorithm" for method in method_circuits]
    if colors is None:
        colors = {metrics[0]: 'blue', metrics[1]: 'orange'}
    
    # Create figure and axes
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, len(method_circuits))
    
    all_axes = []
    
    # Determine global y-axis limits for consistency
    y1_min, y1_max = float('inf'), float('-inf')
    y2_min, y2_max = float('inf'), float('-inf')
    
    # First pass to determine global y limits
    for method in method_circuits:
        method_data = data[data['method_circuit'] == method]
        
        for j, metric in enumerate(metrics):
            metric_data = method_data[method_data['metric'] == metric]
            grouped = metric_data.groupby('iter')
            mean_values = grouped['value'].mean()
            std_values = grouped['value'].std()
            
            y = mean_values.values
            yerr = std_values.values
            
            if j == 0:  # First metric (left axis)
                y1_min = min(y1_min, (y - yerr).min())
                y1_max = max(y1_max, (y + yerr).max())
            else:  # Second metric (right axis)
                y2_min = min(y2_min, (y - yerr).min())
                y2_max = max(y2_max, (y + yerr).max())
    
    # Add some padding to the limits
    y1_range = y1_max - y1_min
    y2_range = y2_max - y2_min
    y1_min -= 0.05 * y1_range
    y1_max += 0.05 * y1_range
    y2_min -= 0.05 * y2_range
    y2_max += 0.05 * y2_range
    
    # Create a subplot for each method_circuit
    for i, method in enumerate(method_circuits):
        # Filter data for this method
        method_data = data[data['method_circuit'] == method]
        
        # Create subplot with twin y-axis
        ax1 = fig.add_subplot(gs[i])
        ax2 = ax1.twinx()
        all_axes.append((ax1, ax2))
        
        # Plot first metric on left axis
        metric1_data = method_data[method_data['metric'] == metrics[0]]
        grouped = metric1_data.groupby('iter')
        mean_values = grouped['value'].mean()
        std_values = grouped['value'].std()
        
        x = mean_values.index
        y = mean_values.values
        yerr = std_values.values
        
        ax1.plot(x, y, color=colors[metrics[0]], label=metrics[0])
        ax1.fill_between(x, y-yerr, y+yerr, color=colors[metrics[0]], alpha=0.2)
        
        # Plot second metric on right axis
        metric2_data = method_data[method_data['metric'] == metrics[1]]
        grouped = metric2_data.groupby('iter')
        mean_values = grouped['value'].mean()
        std_values = grouped['value'].std()
        
        x = mean_values.index
        y = mean_values.values
        yerr = std_values.values
        
        ax2.plot(x, y, color=colors[metrics[1]], label=metrics[1])
        ax2.fill_between(x, y-yerr, y+yerr, color=colors[metrics[1]], alpha=0.2)
        
        # Set common y limits for consistency
        ax1.set_ylim(y1_min, y1_max)
        ax2.set_ylim(y2_min, y2_max)
        
        # Set title
        ax1.set_title(titles[i])
        
        # Set x axis label (only on the bottom)
        ax1.set_xlabel(x_labels[i])
        
        # Set tick colors to black
        ax1.tick_params(axis='both', colors='black')
        ax2.tick_params(axis='both', colors='black')
        
        # Implementation of requirement 1: Show only left axis in leftmost plot and right axis in rightmost plot
        # For middle plots, both axes are hidden
        if i == 0:  # Leftmost plot
            ax1.set_ylabel(left_y_labels[i], color='black')
            ax2.set_yticklabels([])  # Hide right y-axis tick labels
            ax2.tick_params(axis='y', length=0)  # Hide right y-axis ticks
        elif i == len(method_circuits) - 1:  # Rightmost plot
            ax2.set_ylabel(right_y_labels[i], color='black')
            ax1.set_yticklabels([])  # Hide left y-axis tick labels
            ax1.tick_params(axis='y', length=0)  # Hide left y-axis ticks
        else:  # Middle plots
            ax1.set_yticklabels([])  # Hide left y-axis tick labels
            ax2.set_yticklabels([])  # Hide right y-axis tick labels
            ax1.tick_params(axis='y', length=0)  # Hide left y-axis ticks
            ax2.tick_params(axis='y', length=0)  # Hide right y-axis ticks
    
    # Create unified legend at the top
    custom_lines = [
        Line2D([0], [0], color=colors[metrics[0]], lw=2),
        Line2D([0], [0], color=colors[metrics[1]], lw=2)
    ]
    
    # # Create a single legend above the plots
    # fig.legend(
    #     custom_lines, 
    #     metrics, 
    #     loc='upper center', 
    #     bbox_to_anchor=(0.5, 1.02), 
    #     fancybox=True, 
    #     shadow=False, 
    #     ncol=2
    # )
    
    plt.tight_layout()
    # Adjust the layout to accommodate the legend
    plt.subplots_adjust(top=0.85)
    
    return fig



def create_dual_axis_plots_for_indices(data, idx_list, method_circuit, metrics,
                                     x_labels=None, left_y_labels=None, right_y_labels=None,
                                     titles=None, colors=None, figsize=(18, 5)):
    """
    Create dual y-axis plots for different indices of a quantum algorithm.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the data with columns 'idx', 'metric', 'iter', 'value'
    idx_list : list
        List of idx values to plot in order
    method_circuit : str
        The method_circuit to filter data for
    metrics : list
        List of metrics to plot (should be length 2)
    x_labels : list, optional
        Custom x-axis labels for each subplot
    left_y_labels : list, optional
        Custom left y-axis labels for each subplot
    right_y_labels : list, optional
        Custom right y-axis labels for each subplot
    titles : list, optional
        Custom titles for each subplot
    colors : dict, optional
        Custom colors for each metric
    figsize : tuple, optional
        Figure size (width, height)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    if len(metrics) != 2:
        raise ValueError("Exactly two metrics required for dual y-axis plots")
    
    # Filter data for the specific method_circuit
    filtered_data = data[data['method_circuit'] == method_circuit]
    filtered_data = filtered_data[filtered_data['idx'].isin(idx_list)]
    filtered_data = filtered_data[filtered_data['metric'].isin(metrics)]
    
    # Set default values if not provided
    if x_labels is None:
        x_labels = ['iter'] * len(idx_list)
    if left_y_labels is None:
        left_y_labels = [metrics[0]] * len(idx_list)
    if right_y_labels is None:
        right_y_labels = [metrics[1]] * len(idx_list)
    if titles is None:
        titles = [f"idx = {idx}" for idx in idx_list]
    if colors is None:
        colors = {metrics[0]: '#1f77b4', metrics[1]: '#ff7f0e'}  # Blue and orange
    
    # Create figure and axes
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, len(idx_list))
    
    all_axes = []
    
    # Determine global y-axis limits for consistency
    y1_min, y1_max = float('inf'), float('-inf')
    y2_min, y2_max = float('inf'), float('-inf')
    
    # First pass to determine global y limits
    for idx in idx_list:
        idx_data = filtered_data[filtered_data['idx'] == idx]
        
        for j, metric in enumerate(metrics):
            metric_data = idx_data[idx_data['metric'] == metric]
            if metric_data.empty:
                continue
                
            grouped = metric_data.groupby('iter')
            mean_values = grouped['value'].mean()
            
            # Some datasets might not have std values, so we need to handle that
            try:
                std_values = grouped['value'].std()
                std_values.fillna(0, inplace=True)  # Replace NaN with 0
            except:
                std_values = pd.Series(0, index=mean_values.index)
            
            y = mean_values.values
            yerr = std_values.values
            
            if j == 0:  # First metric (left axis)
                if len(y) > 0:
                    y_min = (y - yerr).min() if len(yerr) > 0 else y.min()
                    y_max = (y + yerr).max() if len(yerr) > 0 else y.max()
                    y1_min = min(y1_min, y_min)
                    y1_max = max(y1_max, y_max)
            else:  # Second metric (right axis)
                if len(y) > 0:
                    y_min = (y - yerr).min() if len(yerr) > 0 else y.min()
                    y_max = (y + yerr).max() if len(yerr) > 0 else y.max()
                    y2_min = min(y2_min, y_min)
                    y2_max = max(y2_max, y_max)
    
    # Handle case where no data was found
    if y1_min == float('inf') or y1_max == float('-inf'):
        y1_min, y1_max = 0, 1
    if y2_min == float('inf') or y2_max == float('-inf'):
        y2_min, y2_max = 0, 1
    
    # Add some padding to the limits
    y1_range = y1_max - y1_min
    y2_range = y2_max - y2_min
    y1_padding = 0.05 * y1_range if y1_range > 0 else 0.5
    y2_padding = 0.05 * y2_range if y2_range > 0 else 0.5
    y1_min -= y1_padding
    y1_max += y1_padding
    y2_min -= y2_padding
    y2_max += y2_padding
    
    # Create a subplot for each idx
    for i, idx in enumerate(idx_list):
        idx_data = filtered_data[filtered_data['idx'] == idx]
        
        # Create subplot with twin y-axis
        ax1 = fig.add_subplot(gs[i])
        ax2 = ax1.twinx()
        all_axes.append((ax1, ax2))
        
        # Plot metrics
        for j, metric in enumerate(metrics):
            ax = ax1 if j == 0 else ax2
            
            metric_data = idx_data[idx_data['metric'] == metric]
            if metric_data.empty:
                continue
                
            grouped = metric_data.groupby('iter')
            mean_values = grouped['value'].mean()
            
            # Some datasets might not have std values, so we need to handle that
            try:
                std_values = grouped['value'].std()
                std_values.fillna(0, inplace=True)  # Replace NaN with 0
            except:
                std_values = pd.Series(0, index=mean_values.index)
            
            x = mean_values.index
            y = mean_values.values
            yerr = std_values.values
            
            ax.plot(x, y, color=colors[metric], label=metric)
            
            # Only add fill_between if we have error data
            if np.any(yerr > 0):
                ax.fill_between(x, y-yerr, y+yerr, color=colors[metric], alpha=0.2)
        
        # Set common y limits for consistency
        ax1.set_ylim(y1_min, y1_max)
        ax2.set_ylim(y2_min, y2_max)
        
        # Set title if provided
        if titles and i < len(titles):
            ax1.set_title(titles[i])
        
        # Set x axis label (only on the bottom)
        ax1.set_xlabel(x_labels[i])
        
        # Set tick colors to black
        ax1.tick_params(axis='both', colors='black')
        ax2.tick_params(axis='both', colors='black')
        
        # Implementation of requirement: Show only left axis in leftmost plot and right axis in rightmost plot
        if i == 0:  # Leftmost plot
            ax1.set_ylabel(left_y_labels[i], color='black')
            ax2.set_yticklabels([])  # Hide right y-axis tick labels
            ax2.tick_params(axis='y', length=0)  # Hide right y-axis ticks
        elif i == len(idx_list) - 1:  # Rightmost plot
            ax2.set_ylabel(right_y_labels[i], color='black')
            ax1.set_yticklabels([])  # Hide left y-axis tick labels
            ax1.tick_params(axis='y', length=0)  # Hide left y-axis ticks
        else:  # Middle plots
            ax1.set_yticklabels([])  # Hide left y-axis tick labels
            ax2.set_yticklabels([])  # Hide right y-axis tick labels
            ax1.tick_params(axis='y', length=0)  # Hide left y-axis ticks
            ax2.tick_params(axis='y', length=0)  # Hide right y-axis ticks
    
    # Create unified legend at the top
    custom_lines = [
        Line2D([0], [0], color=colors[metrics[0]], lw=2),
        Line2D([0], [0], color=colors[metrics[1]], lw=2)
    ]
    
    # # Create a single legend above the plots
    # fig.legend(
    #     custom_lines, 
    #     metrics, 
    #     loc='upper center', 
    #     bbox_to_anchor=(0.5, 1.05), 
    #     fancybox=True, 
    #     shadow=False, 
    #     ncol=2
    # )
    
    plt.tight_layout()
    # Adjust the layout to accommodate the legend
    plt.subplots_adjust(top=0.85)
    
    return fig