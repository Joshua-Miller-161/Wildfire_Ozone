import sys
sys.dont_write_bytecode = True
import os
os.environ['USE_PYGEOS'] = '0'
import numpy as np
import yaml
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize, LogNorm, FuncNorm
import joblib
import pickle

import geopandas as gpd
import fiona

sys.path.append(os.getcwd())
from data_utils.preprocessing_funcs import UnScale
from data_utils.rf_data_formatters import NaiveRFDataLoader
from ml.ml_utils import NameModel, ParseModelName
from misc.misc_utils import SavePredData
#====================================================================
def TrainNaiveRF(config_path, data_config_path, model_save_path=None):
    #----------------------------------------------------------------
    ''' Get data from config '''

    with open(config_path, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    assert (config['MODEL_TYPE'] == 'RF'), "'MODEL_TYPE' must be 'RF'. Got: "+config['MODEL_TYPE']
    model_type   = config['MODEL_TYPE']
    num_trees_rf = config['HYPERPARAMETERS']['rf_hyperparams_dict']['num_trees']
    use_xgboost  = config['HYPERPARAMETERS']['rf_hyperparams_dict']['use_xgboost']
    max_depth    = config['HYPERPARAMETERS']['rf_hyperparams_dict']['max_depth']
    subsample    = config['HYPERPARAMETERS']['rf_hyperparams_dict']['subsample']
    device       = config['HYPERPARAMETERS']['rf_hyperparams_dict']['device']
    #----------------------------------------------------------------
    ''' Get data '''

    x_train_df, x_test_df, y_train_df, y_test_df = NaiveRFDataLoader(config_path, data_config_path)
    #print("y_train_df.shape", np.shape(y_train_df.values), np.shape(y_train_df.values.ravel()))
    
    print("}}}}}}}}}}")
    print("}}}}}}}}}}")
    print("}}}}}}}}}}")
    print("}}}}}}}}}}")
    print("xgboostingggg", use_xgboost)
    print("}}}}}}}}}}")
    print("}}}}}}}}}}")
    print("}}}}}}}}}}")
    print("}}}}}}}}}}")
    #----------------------------------------------------------------
    ''' Train & save model '''
    if (model_type == 'RF'):
        if use_xgboost:

            DM_train = xgb.DMatrix(data=x_train_df, label=y_train_df)
            DM_test  = xgb.DMatrix(data=x_test_df, label=y_test_df)

            del(x_train_df)
            del(y_train_df)
            del(x_test_df)
            del(y_test_df)

            param_dict = {"objective": "reg:squarederror",
                          "booster": "gbtree",
                          "colsample_bynode": 0.8,
                          "learning_rate": 1, # DO NOT CHANGE, ENSURES XGBOOST MAKES A RANDOM FOREST
                          "max_depth": max_depth,
                          "num_parallel_tree": num_trees_rf,
                          "subsample": subsample,
                          "tree_method": "hist",
                          "device": device}
            
            evals_result = {}
            model = xgb.train(param_dict,
                             dtrain=DM_train,
                             num_boost_round=1, # DO NOT CHANGE, ENSURES XGBOOST MAKES A RANDOM FOREST
                             evals=[(DM_train, "Train"), (DM_test, "Test")],
                             evals_result=evals_result,
                             verbose_eval=True)
            
            train_error = evals_result["Train"]["rmse"]
            test_error = evals_result["Test"]["rmse"]

            if not (model_save_path == None):
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                model_name = NameModel(config_path)
                print('model_name', model_name)
                if model_name == None:
                    model_name = NameModel(config_path)
                joblib.dump(model, os.path.join(model_save_path, model_name))

        else: 
            rfr = RandomForestRegressor(n_estimators=num_trees_rf, 
                                        max_depth=24)

            print("y_train_df.shape", np.shape(y_train_df.values), np.shape(y_train_df.values.ravel()))
            rfr.fit(x_train_df, y_train_df.values.ravel())

            if not (model_save_path == None):
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                model_name = NameModel(config_path)
                print(' >> model_name', model_name)
                joblib.dump(rfr, open(os.path.join(model_save_path, model_name), 'wb'))
                print(' >> Saved', os.path.join(model_save_path, model_name)) 
#====================================================================
def TestNaiveRF(config_path, data_config_path, model_name, model_pred_path=None):
    #----------------------------------------------------------------
    ''' Get info from model name '''
    info, param_dict = ParseModelName(model_name)
    print("param_dict:", param_dict)

    #----------------------------------------------------------------
    ''' Get data from config '''

    with open(config_path, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    model_folder  = config['MODEL_SAVE_PATH']
    figure_folder = config['FIG_SAVE_PATH']

    if (param_dict['MODEL_TYPE'] == 'XGBRF'):
        config['MODEL_TYPE'] = 'RF'
        config['HYPERPARAMETERS']['rf_hyperparams_dict']['use_xgboost'] = True
    else:
        config['MODEL_TYPE']   = param_dict['MODEL_TYPE']
    config['REGION']       = param_dict['REGION']
    config['RF_OFFSET']    = param_dict['RF_OFFSET']
    config['HISTORY_DATA'] = param_dict['HISTORY_DATA']
    config['TARGET_DATA']  = param_dict['TARGET_DATA']

    with open(config_path, 'w') as config_file:
        yaml.dump(config, config_file)
    #----------------------------------------------------------------
    ''' Load model '''

    trained_rfr = 69
    folders = os.listdir(model_folder)
    if ('.DS_Store' in folders):
        folders.remove('.DS_Store')

    for folder in folders:
        for root, dirs, files in os.walk(os.path.join(model_folder, folder)):
            for name in files:
                #print("root=",root, ", name=", name)
                if (model_name in name):
                    if name.endswith('.joblib'):
                        print(" >> ")
                        print(" >> ", os.path.join(root, name))
                        trained_rfr = joblib.load(os.path.join(root, name))
                        print(" >> Loaded", os.path.join(root, name))
                    elif name.endswith('.pkl'):
                        trained_rfr = pickle.load(open(os.path.join(root, name), 'rb'))
                        print(" >> Loaded", os.path.join(root, name))
    #----------------------------------------------------------------
    ''' Get data '''

    x_train_df, x_test_df, y_train_df, y_test_df, x_train_orig_shape, x_test_orig_shape, y_train_orig_shape, y_test_orig_shape = NaiveRFDataLoader(config_path, data_config_path, 
                                                                                                                                                   return_shapes=True, shuffle=True)

    del(x_train_df)
    del(y_train_df)

    raw_ozone = UnScale(y_test_df['ozone'], 'data_utils/scale_files/ozone_standard.json').reshape(y_test_orig_shape[:-1])
    lon       = UnScale(x_test_df['lon'], 'data_utils/scale_files/lon_minmax.json').reshape(x_test_orig_shape[:-1])
    lat       = UnScale(x_test_df['lat'], 'data_utils/scale_files/lat_minmax.json').reshape(x_test_orig_shape[:-1])
    time      = UnScale(x_test_df['time'], 'data_utils/scale_files/time_minmax.json').reshape(x_test_orig_shape[:-1])
    
    print(" >> x_test :", x_test_df.shape)
    print(" >> y_test :", y_test_df.shape)
    print("____________________________________________________________")
    print(" >> Testing model type:", config['MODEL_TYPE'])
    print("____________________________________________________________")
    # print('UNSCALED OZONE', np.shape(raw_ozone), raw_ozone)
    # print("+++++++++++++++++++++++++++++++")
    print('UNSCALED LON', np.shape(lon), lon)
    print("+++++++++++++++++++++++++++++++")
    print("UNSCALED LAT", np.shape(lat), lat)
    # print("+++++++++++++++++++++++++++++++")
    # print("UNSCALED TIME", np.shape(time), time)
    #----------------------------------------------------------------
    ''' Test model '''
    y_pred = 69
    if ('XGB' in model_name):
        y_pred = np.asarray(trained_rfr.predict(xgb.DMatrix(data=x_test_df)))
    else:
        y_pred = trained_rfr.predict(x_test_df)
    
    y_pred = UnScale(np.squeeze(y_pred), 'data_utils/scale_files/ozone_standard.json').reshape(y_test_orig_shape[:-1])

    print(' >> y_pred:', np.shape(y_pred))
    print("____________________________________________________________")
    
    #----------------------------------------------------------------
    ''' Evaluate model '''
    mse = np.mean(np.square(np.subtract(raw_ozone, y_pred)))

    if not ('XGB' in model_name):
        model_ranks = pd.Series(trained_rfr.feature_importances_,
                                index=x_test_df.columns,
                                name="Importance").sort_values(ascending=True, inplace=False) 
    
    print(" >> mse:", mse)
    print("____________________________________________________________")
    print(" >> model source:", os.path.join(root, model_name))
    print("____________________________________________________________")
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
    try:
        world = gpd.read_file("/Users/joshuamiller/Documents/Lancaster/Data/ne_110m_land/ne_110m_land.shp")
    except fiona.errors.DriverError:
        world = gpd.read_file("/content/drive/MyDrive/Colab_Notebooks/Data/ne_110m_land/ne_110m_land.shp")
    
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
    if not ('XGB' in model_name):
        ax=model_ranks.plot(kind='bar', ax=ax_feat, rot=45)

    ax_feat.text(0, .6, 
                 param_dict['MODEL_TYPE_LONG']+'\n'+config['REGION']+'\nMSE: '+str(round(mse, 10))+'\n'+str(config['RF_OFFSET'])+' days ahead', 
                 fontsize=12, fontweight='bold')
    
    model_name = model_name.split('.')[0]

    fig.savefig(os.path.join(figure_folder, model_name+'.pdf'), bbox_inches=None, pad_inches=0)

    #----------------------------------------------------------------
    ''' Save predictions '''
    
    SavePredData(config_path, model_name, y_pred, raw_ozone, time_axis)
    
    print(' >> Saved predictions:', os.path.join(model_pred_path, model_name+".npy"))
    print("____________________________________________________________")
    #----------------------------------------------------------------
    plt.show()
#====================================================================