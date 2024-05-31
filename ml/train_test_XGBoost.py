import sys
sys.dont_write_bytecode = True
import os
os.environ['USE_PYGEOS'] = '0'
import numpy as np
import yaml
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize, LogNorm, FuncNorm
from joblib import dump, load
import geopandas as gpd

sys.path.append(os.getcwd())
from data_utils.extraction_funcs import Extract_netCDF4
from data_utils.preprocessing_funcs import UnScale
from data_utils.rf_data_formatters import NaiveRFDataLoader
from ml.ml_utils import NameModel
#====================================================================
def TrainNaiveXGBoost(config_path, data_config_path, model_save_path='/Users/joshuamiller/Documents/Lancaster/SavedModels/GBM'):
    #----------------------------------------------------------------
    ''' Get data from config '''

    with open(config_path, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    assert (config['MODEL_TYPE'] == 'GBM'), "'MODEL_TYPE' must be 'GBM'. Got: "+config['MODEL_TYPE']
    model_type            = config['MODEL_TYPE']
    num_boost_round       = config['HYPERPARAMETERS']['gb_hyperparams_dict']['num_boost_round']
    max_depth             = config['HYPERPARAMETERS']['gb_hyperparams_dict']['max_depth']
    early_stopping_rounds = config['HYPERPARAMETERS']['gb_hyperparams_dict']['early_stopping_rounds']
    subsample             = config['HYPERPARAMETERS']['gb_hyperparams_dict']['subsample']
    #----------------------------------------------------------------
    ''' Get data '''

    x_train_df, x_test_df, y_train_df, y_test_df = NaiveRFDataLoader(config_path, data_config_path)
    #----------------------------------------------------------------
    ''' Train model '''

    DM_train = xgb.DMatrix(data=x_train_df, label=y_train_df)
    DM_test  = xgb.DMatrix(data=x_test_df, label=y_test_df)

    del(x_train_df)
    del(y_train_df)
    del(x_test_df)
    del(y_test_df)

    param_dict={'objective': 'reg:squarederror', # fixed. pick an objective function for Regression. 
                'max_depth': max_depth,
                'subsample': subsample,
                'eta': 0.3,
                'min_child_weight': 1,
                'colsample_bytree': 1,
                'gamma': 0,
                'reg_alpha': 0.1,
                'reg_lambda': 1,
                'eval_metric': 'mse', # fixed. picked a evaluation metric for Regression.
                'gpu_id': 0, 
                'tree_method': 'gpu_hist' # XGBoost's built-in GPU support to use Google Colab's GPU
                }
    
    evals_result = {}
    model = xgb.train(param_dict,
                      dtrain=DM_train,
                      num_boost_round=num_boost_round,
                      evals=[(DM_train, "Train"), (DM_test, "Test")],
                      early_stopping_rounds=early_stopping_rounds,
                      evals_result=evals_result,
                      verbose_eval=True)
    
    train_error = evals_result["Train"]["mse"]
    test_error = evals_result["Test"]["mse"]

    #----------------------------------------------------------------
    ''' Save model '''
    model_name = NameModel(config_path)

    dump(os.path.join(model_save_path, model_name))
#====================================================================
def TestNaiveXGBoost(config_path, data_config_path, model_name, model_folder='/Users/joshuamiller/Documents/Lancaster/SavedModels'):
    #----------------------------------------------------------------
    ''' Get data from config '''

    with open(config_path, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    figure_folder = config['FIG_SAVE_PATH']
    #----------------------------------------------------------------
    ''' Load model '''
    model = 69
    folders = os.listdir(model_folder)
    if ('.DS_Store' in folders):
        folders.remove('.DS_Store')

    for folder in folders:
        for root, dirs, files in os.walk(os.path.join(model_folder, folder)):
            for name in files:
                #print("root=",root, ", name=", name)
                if (model_name in name):
                    if name.endswith('.pkl'):

                        model = load(os.path.join(root, name))
    #----------------------------------------------------------------
    ''' Get data '''
    x_train_df, x_test_df, y_train_df, y_test_df, x_train_orig_shape, x_test_orig_shape, y_train_orig_shape, y_test_orig_shape = NaiveRFDataLoader(config_path, data_config_path, return_shapes=True)

    del(x_train_df)
    del(y_train_df)

    raw_ozone = UnScale(y_test_df['ozone'], 'data_utils/scale_files/ozone_standard.json').reshape(y_test_orig_shape[:-1])
    lon       = UnScale(x_test_df['lon'], 'data_utils/scale_files/lon_minmax.json').reshape(x_test_orig_shape[:-1])
    lat       = UnScale(x_test_df['lat'], 'data_utils/scale_files/lat_minmax.json').reshape(x_test_orig_shape[:-1])
    time      = UnScale(x_test_df['time'], 'data_utils/scale_files/time_minmax.json').reshape(x_test_orig_shape[:-1])
    
    print('UNSCALED OZONE', np.shape(raw_ozone), raw_ozone)
    print("+++++++++++++++++++++++++++++++")
    print('UNSCALED LON', np.shape(lon), lon)
    print("+++++++++++++++++++++++++++++++")
    print("UNSCALED LAT", np.shape(lat), lat)
    print("+++++++++++++++++++++++++++++++")
    print("UNSCALED TIME", np.shape(time), time)
    #----------------------------------------------------------------
    ''' Test model '''

    y_pred = model.predict(x_test_df)
    
    y_pred = UnScale(np.squeeze(y_pred), 'data_utils/scale_files/ozone_standard.json').reshape(y_test_orig_shape[:-1])

    print("_________________________________")
    print(np.shape(y_pred))
    print("_________________________________")

    mse = np.mean(np.square(np.subtract(raw_ozone, y_pred)))

    model_ranks = pd.Series(model.feature_importances_,
                            index=x_test_df.columns,
                            name="Importance").sort_values(ascending=True, inplace=False) 
    
    print("mse:", mse)
    #print("model_ranks:", model_ranks)

    #----------------------------------------------------------------
    ''' Plot '''
    fig = plt.figure(layout="constrained", figsize=(10, 7))
    gs = GridSpec(3, 3, figure=fig, wspace=.05, hspace=.05)
    
    ax_orig1 = fig.add_subplot(gs[1, 0])
    ax_pred1 = fig.add_subplot(gs[1, 1])
    ax_orig2 = fig.add_subplot(gs[2, 0])
    ax_pred2 = fig.add_subplot(gs[2, 1])
    ax_time  = fig.add_subplot(gs[0, :])
    ax_feat  = fig.add_subplot(gs[1:, 2])

    idx = np.random.randint(0, np.shape(y_pred)[0], size=2)
    idx1 = idx[0]
    idx2 = idx[1]
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Plot average over time
    time_axis      = np.mean(time, axis=(1, 2))
    raw_ozone_mean = np.mean(raw_ozone, axis=(1,2))
    y_pred_mean    = np.mean(y_pred, axis=(1,2))

    ax_time.axvline(time_axis[idx1], color='r')
    ax_time.axvline(time_axis[idx2], color='r')

    ax_time.scatter(time_axis, raw_ozone_mean, s=15,color='black', marker='x', label='Data')
    ax_time.scatter(time_axis, y_pred_mean, s=20, facecolors='none', edgecolors='blue', marker='o', label='Prediction')

    ax_time.legend()
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Plot 2D maps
    world = gpd.read_file("/Users/joshuamiller/Documents/Lancaster/Data/ne_110m_land/ne_110m_land.shp")
    min_lat = np.min(lat)
    max_lat = np.max(lat)
    min_lon = np.min(lon)
    max_lon = np.max(lon)
    axes = [ax_orig1, ax_orig2, ax_pred1, ax_pred2]
    for i in range(len(axes)):
        axes[i].set_xlim(min_lon-5, max_lon+5)
        axes[i].set_ylim(min_lat-5, max_lat+5)
        world.plot(ax=axes[i], facecolor='none', edgecolor='black', linewidth=.5, alpha=1)

        if (i == 0):
            axes[i].set_title("data "+str(int(time_axis[idx1])))
        elif (i == 1):
            axes[i].set_title("data "+str(int(time_axis[idx2])))
        elif (i == 2):
            axes[i].set_title("pred. "+str(int(time_axis[idx1])))
        elif (i == 3):
            axes[i].set_title("pred. "+str(int(time_axis[idx2])))

    min_ozone = np.min([np.min(raw_ozone), np.min(y_pred)])
    max_ozone = np.max([np.max(raw_ozone), np.max(y_pred)])

    norm = Normalize(vmin=min_ozone, vmax=max_ozone)

    lol = ax_orig1.scatter(x=lon[idx1, ...], y=lat[idx1, ...], c=raw_ozone[idx1, ...], norm=norm, cmap='bwr')
    ax_pred1.scatter(x=lon[idx1, ...], y=lat[idx1, ...], c=y_pred[idx1, ...], norm=norm, cmap='bwr')
    ax_orig2.scatter(x=lon[idx2, ...], y=lat[idx2, ...], c=raw_ozone[idx2, ...], norm=norm, cmap='bwr')
    ax_pred2.scatter(x=lon[idx2, ...], y=lat[idx2, ...], c=y_pred[idx2, ...], norm=norm, cmap='bwr')

    cbar_ax = fig.add_axes([0.61, 0.08, 0.03, 0.59])  # Adjust as necessary
    fig.colorbar(lol, cax=cbar_ax)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Plot feature importance
    ax=model_ranks.plot(kind='bar', ax=ax_feat, rot=45)

    ax_feat.text(0, .6, 
                 config['MODEL_TYPE']+'\n'+config['REGION']+'\nMSE: '+str(round(mse, 10))+'\n'+str(config['RF_OFFSET'])+' days ahead', 
                 fontsize=12, fontweight='bold')
    
    model_name = model_name.split('.')[0]

    fig.savefig(os.path.join(figure_folder, model_name+'.pdf'), bbox_inches='tight', pad_inches=0)

    plt.show()