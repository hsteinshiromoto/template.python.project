import pandas as pd
import numpy as np

import matplotlib
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

import bokeh
from bokeh.plotting import figure, output_notebook, show
from bokeh.layouts import gridplot, row
from bokeh.models import HoverTool, CustomJS, Label
from bokeh.models.widgets import CheckboxGroup

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

from itertools import zip_longest 
from copy import copy
from typeguard import typechecked


def heatmap_4d(volume: pd.DataFrame, probabilities: pd.DataFrame
                ,xlabel: str="xlabel", ylabel: str="ylabel", figsize: tuple=(20, 30)):
    """
    Plots a 4-dimensional heatmap, where the colorbar varies in the interval [0, 1]
    and the circle sizes are integers

    Args:
        volume (pd.DataFrame): Pivoted data frame containing integers values
        probabilities (pd.DataFrame): Pivoted data frame containing values in the interval [0, 1]
        xlabel (str, optional): Name of x label. Defaults to "xlabel".
        ylabel (str, optional):  Name of y label. Defaults to "ylabel".
        figsize (tuple, optional): Figure size. Defaults to (20, 30).

    Returns:
        [type]: matplotlib figure objects

    Example:
        >>> nrows = 25
        >>> ncols = 50
        >>> volume = pd.DataFrame(np.random.randint(0, 1000, size=(nrows, ncols)), columns=[f"col_{i}" for i in range(ncols)])
        >>> probabilities = pd.DataFrame(np.random.randn(nrows, ncols), columns=[f"col_{i}" for i in range(ncols)])
        >>> _, _ = heatmap_4d(volume, probabilities, xlabel="Category_1", ylabel="Category_2")

    References:
        [1] https://blogs.oii.ox.ac.uk/bright/2014/08/12/point-size-legends-in-matplotlib-and-basemap-plots/
        [2] https://stackoverflow.com/questions/54545758/create-equal-aspect-square-plot-with-multiple-axes-when-data-limits-are-differ/54555334#54555334
    """
    
    # 1. Figure Object Instantiation
    fig, heatmap = plt.subplots(figsize=figsize)
    divider = make_axes_locatable(heatmap)
    legend = divider.append_axes("bottom", size=1, pad=1)

    # 2. Heatmap    
    # 2.a. Get Labels
    ylabels = volume.index
    xlabels = volume.columns.values
    x, y = np.meshgrid(np.arange(xlabels.shape[0]), np.arange(ylabels.shape[0]))

    # 2.b. Get Values
    volume = volume.values
    probabilities = probabilities.values
    
    # 2.c. Calculate List of Radii, Make Circles, and Plot
    radii_list = volume/volume.max()/2
    circles = [plt.Circle((j,i), radius=r) for r, j, i in zip(radii_list.flat, x.flat, y.flat)]
    col = PatchCollection(circles, array=probabilities.flatten(), cmap='coolwarm', edgecolors='k', linewidth=2)
    heatmap.add_collection(col)

    heatmap.set(xticks=np.arange(xlabels.shape[0]), yticks=np.arange(ylabels.shape[0])
                ,xticklabels=xlabels, yticklabels=ylabels)

    heatmap.set_xticks(np.arange(xlabels.shape[0]+1)-0.5, minor=True)
    heatmap.set_yticks(np.arange(ylabels.shape[0]+1)-0.5, minor=True)
    heatmap.grid(which='major', linestyle=":")
    heatmap.set_ylabel(ylabel)
    heatmap.set_xlabel(xlabel)

    heatmap.axes.set_aspect('equal')

    # 3. Legend
    # 3.a. Setup Ticks
    leg_xticks = np.arange(xlabels.shape[0])
    leg_yticks = range(2)
    
    # 3.c. Setup Tick Labels
    min_volume = min([volume for volume in volume.flatten() if volume>0])
    leg_xticklabels = np.linspace(min_volume, max(volume.flatten()), len(leg_xticks), dtype=int)
    leg_yticklabels = [0, 1]
    
    # 3.d. Calculate Radii List Statistical Summary
    radii_list_summary = list(np.percentile(radii_list.flatten(), [25, 50, 75]))
    iqr = max(radii_list_summary) - min(radii_list_summary)
    leg_radii_list = copy(radii_list_summary)
    leg_radii_list.append(max(min(radii_list_summary) - 1.5*iqr, min(radii_list.flat)))
    leg_radii_list.append(min(max(radii_list_summary) + 1.5*iqr, max(radii_list.flat)))
    leg_radii_list = sorted(leg_radii_list)
    
    # 3.e. Calculate Volume List Statistical Summary
    vol_summary = list(np.percentile(volume.flatten(), [25, 50, 75]))
    iqr = max(vol_summary) - min(vol_summary)
    leg_vol_stats = copy(vol_summary)
    leg_vol_stats.append(max(min(vol_summary) - 1.5*iqr, min(volume.flat)))
    leg_vol_stats.append(min(max(vol_summary) + 1.5*iqr, max(volume.flat)))
    leg_vol_stats = sorted(leg_vol_stats)
    
    # 3.e. Calculate What Volumes in the Statistical Summary is Closest to the the x tick labels
    leg_vol_idx = dict(zip_longest(leg_xticklabels, leg_xticks))
    leg_vol_list = [leg_xticklabels[(np.abs(leg_xticklabels - volume)).argmin()] for volume in leg_vol_stats]
    
    # 3.f. Get Position for the Circles, and Plot THem
    leg_circle_pos = [leg_vol_idx[item] for item in leg_vol_list]
    leg_circle_pos = sorted(leg_circle_pos)
    legend_circles = [plt.Circle((i, 0.5), radius=r) for r, i in zip(leg_radii_list, leg_circle_pos)]
    legend_col = PatchCollection(legend_circles, edgecolors='k', linewidth=2)
    legend.add_collection(legend_col)

    # Adjust x labels so that only the plotted circles will have an x tick
    xlabels = [label if label in leg_vol_list else "" for label in leg_xticklabels]
    legend.set(xticks=leg_xticks, yticks=leg_yticks, xticklabels=xlabels, yticklabels=[])
    legend.set_xticks(np.arange(len(leg_xticklabels)+2)-0.5, minor=True)
    legend.set_yticks(np.arange(len(leg_yticklabels)+1)-0.5, minor=True)

    # 3.g. Format Plot
    legend.set_xlabel("Volume")
    legend.axes.set_aspect('equal')
    legend.spines['right'].set_visible(False)
    legend.spines['left'].set_visible(False)
    legend.spines['top'].set_visible(False)
    legend.spines['bottom'].set_visible(False)
    legend.tick_params(axis=u'both', which=u'both',length=0)

    # 4. Setup Heatmap Colorbar
    axins = inset_axes(heatmap, width="1%", height="100%", loc='upper right', bbox_to_anchor=(0.05, 0., 1, 1), bbox_transform=heatmap.transAxes, borderpad=1)
    fig.colorbar(col, cax=axins)
    
    fig.tight_layout()
    
    return heatmap, legend


def line_bar_plot(x: str, y_line: str, y_bar: str, data: pd.DataFrame
                ,figsize: tuple=(20, 10), proportions: bool=True):
    """Plot line and bars

    Args:
        x (str): The shared x axis
        y_line (str): Values to be plotted in line
        y_bar (str): Values to be plotted in bars
        data (pd.DataFrame): Dataframe containing the above features
        figsize (tuple, optional): Size of the figure. Defaults to (20, 10).
        proportions (bool, optional): Add proportions each bar to the plot. Defaults to True.

    Returns:
        matplotlib axis objects: line and bar plots
    """
    # Instantiate plotting objects
    fig, (line, bar) = plt.subplots(nrows=2, figsize=figsize)

    x_axis = data[x]
    x_len = x_axis.shape[0]

    # Bar plot
    bar = sns.barplot(x_axis, data[y_bar], ax=bar)

    # Add proportions
    if proportions:
        total = data[y_bar].sum()

        for p in bar.patches:
            height = p.get_height()
            bar.text(p.get_x() + p.get_width()/2., height + 3,
                    f'{100.0*height/total:1.1f}%', ha="center", va="bottom")

    bar.set_xlabel(x)
    bar.set_ylabel(y_bar)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.   
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.   
    bar.get_xaxis().tick_bottom()   
    bar.get_yaxis().tick_left()   

    bar.spines['right'].set_visible(False)
    bar.spines['left'].set_visible(False)
    bar.spines['top'].set_visible(False)
    bar.spines['bottom'].set_visible(False)
    
    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.   
    bar.tick_params(axis='both', which='both',length=0)

    # Line plot
    # Needs to come after bar plot to align x
    y_line_stats = data[y_line].describe()
    line_iqr = y_line_stats["75%"] - y_line_stats["25%"]
    y_line_max = min(y_line_stats["75%"] + 1.5*line_iqr, y_line_stats["max"])
    y_line_min = max(y_line_stats["25%"] - 1.5*line_iqr, y_line_stats["min"])

    line = sns.lineplot(x_axis, data[y_line], linestyle="-", marker="o", label=x, ax=line)
    line.plot(x_axis, x_len*[y_line_max], linestyle=":", color="black", alpha=0.25, label="max")
    line.text(x_axis.values[-1], y_line_max, " max", fontsize=14, color="black", va="center") 
    line.plot(x_axis, x_len*[y_line_stats["mean"]], linestyle="--", color="black", alpha=0.25, label="mean")
    line.text(x_axis.values[-1], y_line_stats["mean"], " mean", fontsize=14, color="black", va="center") 
    line.plot(x_axis, x_len*[y_line_min], linestyle=":", color="black", alpha=0.25, label="min")
    line.text(x_axis.values[-1], y_line_min, " min", fontsize=14, color="black", va="center") 
    line.set_title(f"Plot of {y_line} vs {x}")
    line.set_xlabel("")
    line.set_ylabel(y_line)
    line.set_xticklabels([])
    line.get_legend().remove()

    # Ensure that the axis ticks only show up on the bottom and left of the plot.   
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.   
    line.get_xaxis().tick_bottom()   
    line.get_yaxis().tick_left()   

    line.spines['right'].set_visible(False)
    line.spines['left'].set_visible(False)
    line.spines['top'].set_visible(False)
    line.spines['bottom'].set_visible(False)

    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.   
    line.tick_params(axis='both', which='both',length=0)

    if "interval" in str(data[x].dtype).lower():
        x_axis.values[-1] = f"{x_axis.max().left}+"
        plt.xticks(data[x], x_axis, rotation=45)  

    elif np.issubdtype(data[x].dtype, np.datetime64):
        x_dates = data[x].dt.strftime('%Y-%m-%d').sort_values()
        bar.set_xticklabels(labels=x_dates, rotation=45)

    return line, bar


def dash_line(x: str, y: str, data: pd.DataFrame, figsize: tuple=(1000, 400)
            ,title: str=None, x_axis_label: str=None, y_axis_label: str=None):
    """Create dashboard line plot

    Args:
        x (str): Column in x axis
        y (str): Column in y axis
        data (pd.DataFrame): Data frame contaning the above. It also must contain
                            columns for the max, min and mean of y axis
        figsize (tuple, optional): Width and height of figure. Defaults to (1000, 400).
        title (str, optional): Figure title. Defaults to None.
        x_axis_label (str, optional): X-axis label. Defaults to None.
        y_axis_label (str, optional): Y-axis label. Defaults to None.

    Returns:
        bokeh object: Plotted figure

    References:
        [1] https://www.reddit.com/r/learnpython/comments/8ythxo/how_to_use_checkbox_in_bokeh_with_pandas/

    Example: TODO: 
        >>> show(dash_line())
    """    
    # 1. Define variables

    y_max = f"{y}_max"
    y_min = f"{y}_min"
    y_mean = f"{y}_mean"

    title = title or f"{y} vs {x}"

    tools = "xwheel_pan, ywheel_pan, pan,wheel_zoom,xwheel_zoom,ywheel_zoom,box_zoom,reset,save"

    # 2. Instantiate plot object
    
    if np.issubdtype(data[x].dtype, np.datetime64):
        line = figure(x_axis_type="datetime", plot_width=figsize[0]
                    ,plot_height=figsize[1], title=title
                    ,toolbar_location="above", tools=tools)

    else:
        line = figure(plot_width=figsize[0], plot_height=figsize[1]
                    ,title=title, toolbar_location="above", tools=tools)

    ## 2.1. Add axes labels

    line.xaxis.axis_label = x or x_axis_label
    line.yaxis.axis_label = y or y_axis_label
    
    # 3. Add circles
    line.circle(x=x, y=y, source=data)

    # 4. Add hover tooltip
    ## 4.1. Format the tooltip
    tooltips = [(y, f"@{y}"), (x, f"@{x}")]
    hover = HoverTool(tooltips=tooltips)

    ## 4.2. Add the HoverTool to the figure
    line.add_tools(HoverTool(tooltips=tooltips))

    # 5. Format figure
    ## 5.1. Remove grey frame
    line.outline_line_color = None

    ## 5.2. Remove axis line
    line.xaxis.axis_line_width = 0
    line.yaxis.axis_line_width = 0

    ## 5.3. Remove grid
    line.xgrid.grid_line_color = None
    line.ygrid.grid_line_color = None

    ## 5.4. Remove tick markers
    # line.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
    line.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks

    # line.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
    line.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks

    ## 5.5. Add text
    # mytext = Label(x=data[x].max(), y=data[y].iloc[-1], text=y)
    # line.add_layout(mytext)

    # 6. Create rendering objects for callback
    line_renderer = line.line(x=x, y=y, line_width=2, source=data)
    line_max = line.line(x=x, y=y_max, line_width=1, source=data
                        ,line_dash="dashed", line_alpha=0.25
                        ,line_color="black")
    line_mean = line.line(x=x, y=y_mean, line_width=1, source=data
                        ,line_dash="dotted", line_alpha=0.25
                        ,line_color="black")
    line_min = line.line(x=x, y=y_min, line_width=1, source=data
                        ,line_dash="dashed", line_alpha=0.25
                        ,line_color="black")

    # 7. Create checkbox callback. See [1]
    checkboxes = CheckboxGroup(labels=["max", "mean", "min"], active=[0, 1, 2])
    callback = CustomJS(code="""line_max.visible = false; // same xline passed in from args
                                line_mean.visible = false;
                                line_min.visible = false;
                                // cb_obj injected in by the callback
                                if (cb_obj.active.includes(0)){line_max.visible = true;}
                                if (cb_obj.active.includes(1)){line_mean.visible = true;}
                                if (cb_obj.active.includes(2)){line_min.visible = true;}
                                """,
                        args={'line_max': line_max, 'line_mean': line_mean, 'line_min': line_min})
    checkboxes.js_on_click(callback)
    
    return row(line, checkboxes)


def hist_box(feature: str, data: pd.DataFrame, figsize: tuple=(20, 10)):
    """Plots histogram and box

    Args:
        feature (str): Feature to be plotted
        data (pd.DataFrame): Dataframe containing the feature
        figsize (tuple, optional): Size of figure. Defaults to (20, 10).

    Returns:
        matplotlib axis objects: line and bar plots
    """

    # Instantiate plotting objects
    fig, (hist, box) = plt.subplots(nrows=2, figsize=figsize, sharex=True
                                    ,gridspec_kw={'height_ratios': [0.75, 0.25]})

    hist = sns.histplot(data[feature], ax=hist)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.    
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.    
    hist.get_xaxis().tick_bottom()    
    hist.get_yaxis().tick_left()    
    
    hist.spines['right'].set_visible(False)
    hist.spines['left'].set_visible(False)
    hist.spines['top'].set_visible(False)
    hist.spines['bottom'].set_visible(False)

    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
    hist.tick_params(axis='both', which='both',length=0)
    hist.set_title(f"Plot of the Distribution of {feature}")

    box = sns.boxplot(data[feature], ax=box)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.    
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.    
    box.get_xaxis().tick_bottom()    
    box.get_yaxis().tick_left()    
    
    box.spines['right'].set_visible(False)
    box.spines['left'].set_visible(False)
    box.spines['top'].set_visible(False)
    box.spines['bottom'].set_visible(False)

    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
    box.tick_params(axis='both', which='both',length=0)

    return hist, box