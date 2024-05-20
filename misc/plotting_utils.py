import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import numpy as np
from datetime import datetime
import yaml
#====================================================================
def Plot_O3_Fire(ax, fire_x, fire, O3_x, O3, 
                 fire_color='red', O3_color='blue', alpha=1,
                 fire_ymin=None, fire_ymax=None, O3_ymin=None, O3_ymax=None,
                 num_ticks=None, fire_ylabel=None, O3_ylabel=None, fire_label=None, O3_label=None,
                 draw_line_O3=None, draw_line_fire=None):
    
    if (fire_label==None):
        ax.plot(fire_x, fire, color=fire_color, linestyle='-', linewidth=1, alpha=alpha)
    else:
        ax.plot(fire_x, fire, color=fire_color, linestyle='-', linewidth=1, alpha=alpha, label=fire_label)

    if not (fire_ymin == None and fire_ymax == None):
        ax.set_ylim(fire_ymin, fire_ymax)
        if not (num_ticks == None):
            ax.set_yticks(np.linspace(fire_ymin, fire_ymax, num_ticks))
    if not (num_ticks == None):
        ax.set_yticks(np.linspace(min(fire), max(fire), num_ticks))

    ax.tick_params(axis='y', labelcolor=fire_color)
    ax.set_xlabel('Date')
    if (fire_ylabel==None):
        ax.set_ylabel('Avg. FRP (MW)', color=fire_color)
    else:
        ax.set_ylabel(fire_ylabel, color=fire_color)

    if not(draw_line_fire==None):
        ax.plot(fire_x, np.ones_like(fire_x)*draw_line_fire, color='lightcoral', linewidth=1)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    if (O3_label == None):
        ax2.plot(O3_x, O3, color=O3_color, linestyle='-', linewidth=1, alpha=alpha)
    else:
        print("AHHHHHHHHHHHH")
        ax2.plot(O3_x, O3, color=O3_color, linestyle='-', linewidth=1, alpha=alpha, label=O3_label)

    if not (O3_ymin == None and O3_ymax == None):
        ax2.set_ylim(O3_ymin, O3_ymax)
        if not (num_ticks == None):
            ax2.set_yticks(np.linspace(O3_ymin, O3_ymax, num_ticks))
    if not (num_ticks == None):
        ax2.set_yticks(np.linspace(min(O3), max(O3), num_ticks))

    #ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax.get_yticks())))
    if (O3_ylabel==None):
        ax2.set_ylabel(r'$O_3$'+' '+r'$\left(\frac{mol}{m^2}\right)$', color=O3_color)  # we already handled the x-label with ax
    else:
        ax2.set_ylabel(O3_ylabel, color=O3_color)

    if not(draw_line_O3==None):
        ax2.plot(O3_x, np.ones_like(O3_x)*draw_line_O3, color='black', linewidth=1)
    
    ax2.tick_params(axis='y', labelcolor='blue')

def format_date(x, pos):
    date = mdates.num2date(x)
    if date.month == 1:
        return str(date.year)
    else:
        return date.strftime('%b')

def ShowYearMonth(ax, dates):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(format_date))
    ax.tick_params(axis='x', rotation=60)
    ax.grid(axis='x')
    for year in range(dates[0].year, dates[-1].year + 1):
        january = mdates.date2num(datetime(year, 1, 1))
        ax.axvline(x=january, color='black')

def PlotBoxes(config_path, ax, plot_text=False):
    with open(config_path, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)
    min_lat_0 = config['WHOLE_AREA_BOX']['min_lat']
    max_lat_0 = config['WHOLE_AREA_BOX']['max_lat']
    min_lon_0 = config['WHOLE_AREA_BOX']['min_lon']
    max_lon_0 = config['WHOLE_AREA_BOX']['max_lon']

    min_lat_1 = config['EAST_OCEAN_BOX']['min_lat']
    max_lat_1 = config['EAST_OCEAN_BOX']['max_lat']
    min_lon_1 = config['EAST_OCEAN_BOX']['min_lon']
    max_lon_1 = config['EAST_OCEAN_BOX']['max_lon']

    min_lat_2 = config['WEST_OCEAN_BOX']['min_lat']
    max_lat_2 = config['WEST_OCEAN_BOX']['max_lat']
    min_lon_2 = config['WEST_OCEAN_BOX']['min_lon']
    max_lon_2 = config['WEST_OCEAN_BOX']['max_lon']

    min_lat_3 = config['NORTH_LAND_BOX']['min_lat']
    max_lat_3 = config['NORTH_LAND_BOX']['max_lat']
    min_lon_3 = config['NORTH_LAND_BOX']['min_lon']
    max_lon_3 = config['NORTH_LAND_BOX']['max_lon']

    min_lat_4 = config['SOUTH_LAND_BOX']['min_lat']
    max_lat_4 = config['SOUTH_LAND_BOX']['max_lat']
    min_lon_4 = config['SOUTH_LAND_BOX']['min_lon']
    max_lon_4 = config['SOUTH_LAND_BOX']['max_lon']

    box0 = np.array([[min_lon_0, max_lon_0, max_lon_0, min_lon_0, min_lon_0], 
                    [max_lat_0, max_lat_0, min_lat_0, min_lat_0, max_lat_0]])
    box1 = np.array([[min_lon_1, max_lon_1, max_lon_1, min_lon_1, min_lon_1], 
                    [max_lat_1, max_lat_1, min_lat_1, min_lat_1, max_lat_1]])
    box2 = np.array([[min_lon_2, max_lon_2, max_lon_2, min_lon_2, min_lon_2], 
                    [max_lat_2, max_lat_2, min_lat_2, min_lat_2, max_lat_2]])
    box3 = np.array([[min_lon_3, max_lon_3, max_lon_3, min_lon_3, min_lon_3], 
                    [max_lat_3, max_lat_3, min_lat_3, min_lat_3, max_lat_3]])
    box4 = np.array([[min_lon_4, max_lon_4, max_lon_4, min_lon_4, min_lon_4], 
                    [max_lat_4, max_lat_4, min_lat_4, min_lat_4, max_lat_4]])

    ax.plot(box0[0, :], box0[1, :], 'k-')
    ax.plot(box1[0, :], box1[1, :], 'k-')
    ax.plot(box2[0, :], box2[1, :], 'k-')
    ax.plot(box3[0, :], box3[1, :], 'k-')
    ax.plot(box4[0, :], box4[1, :], 'k-')

    if plot_text:
        ax.text(box0[0, 0]+1, box0[1, 0]-2, 'whole area', fontsize=7)
        ax.text(box1[0, 0]+1, box1[1, 0]-2, 'east ocean', fontsize=7)
        ax.text(box2[0, 0]+1, box2[1, 0]-2, 'west ocean', fontsize=7)
        ax.text(box3[0, 0]+1, box3[1, 0]-2, 'north land', fontsize=7)
        ax.text(box4[0, 0]+1, box4[1, 0]-2, 'south land', fontsize=7)