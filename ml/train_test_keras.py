import sys
sys.dont_write_bytecode = True
import os
os.environ['USE_PYGEOS'] = '0'
import numpy as np
import tensorflow as tf
from tensorflow import keras
import yaml
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize, LogNorm, FuncNorm
import geopandas as gpd

sys.path.append(os.getcwd())
from data_utils.preprocessing_funcs import UnScale
from data_utils.data_loader import DataLoader
from data_utils.prepare_histories_targets import Histories_Targets
from data_utils.train_test_split import Train_Test_Split
from ml.conv_lstm import MakeConvLSTM
from ml.ml_utils import NameModel
#====================================================================
def TrainConvLSTM(config_path, model_save_path='/Users/joshuamiller/Documents/Lancaster/SavedModels/ConvLSTM'):
    #----------------------------------------------------------------
    ''' Check for GPU access '''
    
    print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")
    print(f"TensorFlow version: {tf.__version__}")
    #----------------------------------------------------------------
    ''' Get data from config '''

    with open(config_path, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    assert (config['MODEL_TYPE'] == 'ConvLSTM'), "To use this, 'MODEL_TYPE' must be 'ConvLSTM'. Got: "+str(config['MODEL_TYPE'])

    num_epochs    = config['HYPERPARAMETERS']['convlstm_dict']['epochs']
    batch_size    = config['HYPERPARAMETERS']['convlstm_dict']['batch_size']
    #----------------------------------------------------------------
    ''' Get data '''
    x_data = DataLoader('config.yml', 'data_utils/data_utils_config.yml', 'HISTORY_DATA')
    y_data = DataLoader('config.yml', 'data_utils/data_utils_config.yml', 'TARGET_DATA')

    histories, targets = Histories_Targets('config.yml', x_data, y_data, shuffle=True)

    x_train, x_test, y_train, y_test = Train_Test_Split('config.yml', histories, targets, shuffle=True)

    print("x_train", np.shape(x_train), ", y_train", np.shape(y_train), ", x_test", np.shape(x_test), ", y_test", np.shape(y_test))

    #----------------------------------------------------------------
    ''' Train model '''
    model = MakeConvLSTM(config_path, np.shape(x_train), np.shape(y_train))
    
    model.compile(loss=keras.losses.MeanSquaredError(reduction="sum_over_batch_size", 
                                                     name="MSE"),
                  optimizer=keras.optimizers.Adam(learning_rate=1e-3))
    
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss",
                                                      patience=3, 
                                                      restore_best_weights=True)

    history = model.fit(x=x_train,
                        y=y_train,
                        validation_data=(x_test, y_test),
                        batch_size=batch_size,
                        epochs=num_epochs,
                        verbose=1,
                        callbacks=early_stopping_cb)
    #----------------------------------------------------------------
    ''' Save model '''
    if not (model_save_path==None):
        model_name = NameModel(config_path)
        print('model_name', model_name)
    
        model_json = model.to_json()
        with open(os.path.join(model_save_path, model_name+'.json'), 'w') as json_file:
            json_file.write(model_json)

        model.save_weights(os.path.join(model_save_path, model_name+'.h5'))

    return x_test, y_test, history
#====================================================================
def TestConvLSTM(config_path, model_name, model_folder='/Users/joshuamiller/Documents/Lancaster/SavedModels'):
    #----------------------------------------------------------------
    ''' Get data from config '''

    with open(config_path, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)
    #----------------------------------------------------------------
    ''' Get data '''

    x_data = DataLoader(config_path, 'data_utils/data_utils_config.yml', 'HISTORY_DATA')
    y_data = DataLoader(config_path, 'data_utils/data_utils_config.yml', 'TARGET_DATA')

    histories, targets = Histories_Targets('config.yml', x_data, y_data, shuffle=True)

    del(x_data)
    del(y_data)

    x_train, x_test, y_train, y_test = Train_Test_Split('config.yml', histories, targets, shuffle=True)
    
    del(histories)
    del(targets)

    print("x_train", np.shape(x_train), ", y_train", np.shape(y_train), ", x_test", np.shape(x_test), ", y_test", np.shape(y_test))
    x_test_orig_shape = np.shape(x_test)
    y_test_orig_shape = np.shape(y_test)

    raw_ozone = UnScale(y_test, 'data_utils/scale_files/ozone_standard.json').reshape(y_test_orig_shape[:-1])
    lon       = UnScale(x_test[..., -3], 'data_utils/scale_files/lon_minmax.json').reshape(x_test_orig_shape[:-1])
    lat       = UnScale(x_test[..., -2], 'data_utils/scale_files/lat_minmax.json').reshape(x_test_orig_shape[:-1])
    time      = UnScale(x_test[..., -1], 'data_utils/scale_files/time_minmax.json').reshape(x_test_orig_shape[:-1])
    
    print('UNSCALED OZONE', np.shape(raw_ozone), raw_ozone)
    print("+++++++++++++++++++++++++++++++")
    print('UNSCALED LON', np.shape(lon), lon)
    print("+++++++++++++++++++++++++++++++")
    print("UNSCALED LAT", np.shape(lat), lat)
    print("+++++++++++++++++++++++++++++++")
    print("UNSCALED TIME", np.shape(time), time)

    #----------------------------------------------------------------
    ''' Get trained model '''

    model_architecture = ''
    model_weights = ''
    for root, dirs, files in os.walk(model_folder):
        for name in files:
            if (model_name in name):
                if name.endswith('.json'):
                    model_architecture = os.path.join(root, name)
                elif name.endswith('.h5'):
                    model_weights = os.path.join(root, name)
            
    with open(model_architecture, 'r') as json_file:
        loaded_model_json = json_file.read()

    model = keras.models.model_from_json(loaded_model_json)

    model.load_weights(model_weights)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss=keras.losses.MeanSquaredError(reduction="sum_over_batch_size", name="MSE"))


    #----------------------------------------------------------------
    ''' Test model '''

    y_pred = model.predict(x_test)

    y_pred = UnScale(np.squeeze(y_pred), 'data_utils/scale_files/ozone_standard.json').reshape(y_test_orig_shape[:-1])
                     
    print("_________________________________")
    print("y_pred=", np.shape(y_pred), ", y_data=", np.shape(raw_ozone),
          ", lon=", np.shape(lon), ", lat=", np.shape(lat))
    print("_________________________________")

    mse = np.mean(np.square(np.subtract(raw_ozone, y_pred)))
    print("MSE:", mse)

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
    time_axis      = np.mean(time, axis=(1, 2, 3))
    raw_ozone_mean = np.mean(raw_ozone, axis=(1, 2, 3))
    y_pred_mean    = np.mean(y_pred, axis=(1,2, 3))

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

    lol = ax_orig1.scatter(x=lon[idx1, 0, ...], y=lat[idx1, 0, ...], c=raw_ozone[idx1, ...], norm=norm, cmap='bwr')
    ax_pred1.scatter(x=lon[idx1, 0, ...], y=lat[idx1, 0, ...], c=y_pred[idx1, ...], norm=norm, cmap='bwr')
    ax_orig2.scatter(x=lon[idx2, 0, ...], y=lat[idx2, 0, ...], c=raw_ozone[idx2, ...], norm=norm, cmap='bwr')
    ax_pred2.scatter(x=lon[idx2, 0, ...], y=lat[idx2, 0, ...], c=y_pred[idx2, ...], norm=norm, cmap='bwr')

    cbar_ax = fig.add_axes([0.61, 0.08, 0.03, 0.59])  # Adjust as necessary
    fig.colorbar(lol, cax=cbar_ax)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Plot feature importance
    #ax=model_ranks.plot(kind='bar', ax=ax_feat, rot=45)

    ax_feat.text(0, .6, 
                 config['MODEL_TYPE']+'\n'+config['REGION']+'\nMSE: '+str(round(mse, 10))+'\n'+str(config['RF_OFFSET'])+' days ahead', 
                 fontsize=12, fontweight='bold')
    
    #model_name = model_path.split('/')[-1]

    fig.savefig(os.path.join('Figs', model_name+'.pdf'), bbox_inches=None, pad_inches=0)

    plt.show()
#====================================================================
#x_test, y_test, history = TrainConvLSTM('config.yml')

#TestConvLSTM('config.yml', 'CONVLSTM_reg=SL_f=1_In=OFTUVXYD_Out=O_e=10')