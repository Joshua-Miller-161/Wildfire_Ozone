import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import numpy as np
from datetime import datetime, timedelta
import yaml
#====================================================================
def Plot_O3_Fire(ax, fire_x, fire, O3_x, O3, 
                 fire_color='red', O3_color='blue', fire_linewidth=1, O3_linewidth=1, alpha=1,
                 fire_ymin=None, fire_ymax=None, O3_ymin=None, O3_ymax=None,
                 fire_xmin=None, fire_xmax=None, O3_xmin=None, O3_xmax=None,
                 num_ticks=None, fire_ylabel=None, O3_ylabel=None, fire_label=None, O3_label=None,
                 draw_line_O3=None, draw_line_fire=None, fire_rotation=90, O3_rotation=90, fire_pad=0, O3_pad=0,
                 fire_text='', O3_text='',
                 fontsize=12, bold=False):
    
    if (fire_label==None):
        ax.plot(fire_x, fire, color=fire_color, linestyle='-', linewidth=fire_linewidth, alpha=alpha)
    else:
        ax.plot(fire_x, fire, color=fire_color, linestyle='-', linewidth=fire_linewidth, alpha=alpha, label=fire_label)

    if not (fire_ymin == None and fire_ymax == None):
        ax.set_ylim(fire_ymin, fire_ymax)
        if not (num_ticks == None):
            ax.set_yticks(np.linspace(fire_ymin, fire_ymax, num_ticks))
    if not (num_ticks == None):
        ax.set_yticks(np.linspace(min(fire), max(fire), num_ticks))

    if not (fire_xmin == None and fire_xmax == None):
        ax.set_xlim(fire_xmin, fire_xmax)

    ax.tick_params(axis='y', labelcolor=fire_color)
    #ax.set_xlabel('Date')
    if (fire_ylabel==None):
        if bold:
            ax.set_ylabel(r'$\mathbf{'+'Avg. FRP (MW)'+r'}$', color=fire_color, fontsize=fontsize, rotation=fire_rotation, labelpad=fire_pad)
        else:
            ax.set_ylabel('Avg. FRP (MW)', color=fire_color, fontsize=fontsize, rotation=fire_rotation, labelpad=fire_pad)
    else:
        if bold:
            ax.set_ylabel(r'$\mathbf{'+fire_ylabel+r'}$', color=fire_color, fontsize=fontsize, rotation=fire_rotation, labelpad=fire_pad)
        else:
            ax.set_ylabel(fire_ylabel, color=fire_color, fontsize=fontsize, rotation=fire_rotation, labelpad=fire_pad)

    if not(draw_line_fire==None):
        ax.plot(fire_x, np.ones_like(fire_x)*draw_line_fire, color='lightcoral', linewidth=1)

    if (not fire_text==''):
        if bold:
            if (fire_xmin == None):
                ax.text(0.05*min(fire_x), .7*max(fire), fire_text, fontsize=fontsize, fontweight='bold')
            else:
                ax.text(0.05*fire_xmin, .7*max(fire), fire_text, fontsize=fontsize, fontweight='bold')
        else:
            if (fire_xmin == None):
                ax.text(0.05*min(fire_x), .7*max(fire), fire_text, fontsize=fontsize)
            else:
                ax.text(0.05*fire_xmin, .7*max(fire), fire_text, fontsize=fontsize)
    #plt.setp(ax.xaxis.get_majorticklabels(), rotation=fire_rotation)
    #ax.tick_params(axis='y', rotation=fire_rotation)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    if (O3_label == None):
        ax2.plot(O3_x, O3, color=O3_color, linestyle='-', linewidth=O3_linewidth, alpha=alpha)
    else:
        print("AHHHHHHHHHHHH")
        ax2.plot(O3_x, O3, color=O3_color, linestyle='-', linewidth=O3_linewidth, alpha=alpha, label=O3_label)

    if not (O3_ymin == None and O3_ymax == None):
        ax2.set_ylim(O3_ymin, O3_ymax)
        if not (num_ticks == None):
            ax2.set_yticks(np.linspace(O3_ymin, O3_ymax, num_ticks))
    if not (num_ticks == None):
        ax2.set_yticks(np.linspace(min(O3), max(O3), num_ticks))

    if not (O3_xmin == None and O3_xmax == None):
        ax.set_xlim(O3_xmin, O3_xmax)

    #ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax.get_yticks())))
    if (O3_ylabel==None):
        if bold:
            ax2.set_ylabel(r'$\mathbf{O_3}$'+' '+r'$\mathbf{\left(\frac{mol}{m^2}\right)}$', color=O3_color, fontsize=fontsize, rotation=O3_rotation, labelpad=O3_pad)  # we already handled the x-label with ax
        else:
            ax2.set_ylabel(r'$O_3$'+' '+r'$\left(\frac{mol}{m^2}\right)$', color=O3_color, fontsize=fontsize, rotation=O3_rotation, labelpad=O3_pad)  # we already handled the x-label with ax
    else:
        if bold:
            ax2.set_ylabel(r'$\mathbf{'+O3_ylabel+r'}$', color=O3_color, fontsize=fontsize, rotation=O3_rotation, labelpad=O3_pad)
        else:    
            ax2.set_ylabel(O3_ylabel, color=O3_color, fontsize=fontsize, rotation=O3_rotation, labelpad=O3_pad)

    if not(draw_line_O3==None):
        ax2.plot(O3_x, np.ones_like(O3_x)*draw_line_O3, color='black', linewidth=1)
    
    ax2.tick_params(axis='y', labelcolor='blue')

    #plt.setp(ax2.xaxis.get_majorticklabels(), rotation=O3_rotation)
    #ax2.tick_params(axis='y', rotation=O3_rotation)
#====================================================================
def format_date(x, pos):
    date = mdates.num2date(x)
    if date.month == 1:
        return '\\textbf{' + str(date.year) + '}'
    else:
        return date.strftime('%b')
#====================================================================
def format_date_2(x, pos):
    date = mdates.num2date(x)
    if date.month == 7:
        return r'$\mathbf{' + str(date.year) + r'}$'
    else:
        return ''
#====================================================================
def ShowYearMonth(ax, dates, start_date=datetime(1970, 1, 1), fontsize=12, method=0, rotation=0):
    if not (type(dates[0])==datetime):
        dates = np.asarray([start_date + timedelta(days=int(d)) for d in dates])

    ax.xaxis.set_major_locator(mdates.MonthLocator())

    if (method == 0):
        ax.xaxis.set_major_formatter(FuncFormatter(format_date_2))
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(format_date))

    ax.tick_params(axis='x', rotation=rotation, labelsize=fontsize)
    #ax.grid(axis='x')
    for year in range(dates[366].year, dates[-1].year + 1):
        january = mdates.date2num(datetime(year, 1, 1))
        ax.axvline(x=january, color='gray', linestyle=':')
#====================================================================
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
        ax.text(box0[0, 0]+1, box0[1, 0]-7, 'Whole\narea', fontsize=20, fontweight='bold')
        ax.text(box1[0, 0]+1, box1[1, 0]-7, 'East\nocean', fontsize=20, fontweight='bold')
        ax.text(box2[0, 0]+1, box2[1, 0]-7, 'West\nocean', fontsize=20, fontweight='bold')
        ax.text(box3[0, 0]+1, box3[1, 0]-7, 'North\nland', fontsize=20, fontweight='bold')
        ax.text(box4[0, 0]+1, box4[1, 0]-7, 'South\nland', fontsize=20, fontweight='bold')