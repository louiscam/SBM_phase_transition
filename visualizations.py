import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False

def heatmap_with_values(data, cmap_scale=True, cmap_min=0, cmap_max=0, save_to='', xlabel='b', ylabel='a'):
    '''
    Function to plot the heatmap of an input dataframe, along with the corresponding values.
    Args:
        data: (pandas DataFrame) the dataframe to visualize
        cmap_scale: (Boolean) whether the colorbar should be defined between 0 and 1
        cmap_min: (float) min value for the colormap, only used if cmap_scale=False
        cmap_max: (float) max value for the colormap, only used if cmap_scale=False
        save_to: (String) if non empty string, saves the plot with the provided name
        xlabel: (String) the label for the x axis
        ylabel: (String) the label for the y axis
    Returns:
        void
    '''
    # Shape of the input dataframe
    a = data.shape[0]
    b = data.shape[1]

    # Limits for the extent
    x_start = 0
    x_end = a
    y_start = 0
    y_end = b
    extent = [x_start, x_end, y_start, y_end]

    # Plot heatmap
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    if cmap_scale:
        im = ax.imshow(data, extent=extent, origin='upper', interpolation='none', cmap='rocket', vmin=0, vmax=1)
        color_thresh = 0.5
    else:
        if cmap_max==0 and cmap_min==0:
            im = ax.imshow(data, extent=extent, origin='upper', interpolation='none', cmap='rocket')
            color_thresh = data.min().min()+(data.max().max()-data.min().min())/2
        else:
            im = ax.imshow(data, extent=extent, origin='upper', interpolation='none', cmap='rocket', vmin=cmap_min, vmax=cmap_max)
            color_thresh = cmap_min+(cmap_max-cmap_min)/2
    # Add the text
    jump_x = (x_end - x_start) / (2.0 * data.shape[0])
    jump_y = (y_end - y_start) / (2.0 * data.shape[1])
    x_positions = np.linspace(start=x_start, stop=x_end, num=a, endpoint=False)
    y_positions = np.linspace(start=y_start, stop=y_end, num=b, endpoint=False)
    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            label = data.iloc[a-1-y_index, x_index]
            text_x = x + jump_x
            text_y = y + jump_y
            if (label>color_thresh):
                ax.text(text_x, text_y, label, color='black', ha='center', va='center')
            else:
                ax.text(text_x, text_y, label, color='white', ha='center', va='center')

    # Place ticks at the middle of every pixel
    ax.set_xticks(np.arange(a)+0.5)
    ax.set_yticks(np.arange(b)+0.5)
    # Use input dataframe row and column names as ticks
    ax.set_xticklabels(np.array(data.columns, dtype='str'))
    ax.set_yticklabels(np.array(data.index, dtype='str')[::-1])
    # Define axis name
    ax.xaxis.set_label_position('top') 
    ax.xaxis.set_ticks_position('top') 
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Add colorbar
    fig.colorbar(im)
    
    # Save to pdf
    if save_to != '':
        plt.savefig(save_to, format='png')
    
    plt.show()
    



