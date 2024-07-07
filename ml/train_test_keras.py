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
import fiona

sys.path.append(os.getcwd())
from data_utils.preprocessing_funcs import UnScale
from data_utils.data_loader import DataLoader
from data_utils.prepare_histories_targets import Histories_Targets
from data_utils.train_test_split import Train_Test_Split
from ml.conv_lstm import MakeConvLSTM
from ml.conv import MakeConv
from ml.lstm import MakeLSTM
from ml.rbdn import MakeRBDN
from ml.splitter import MakeSplitter
from ml.denoise3D import MakeDenoise
#from ml.denoise3DOrig import MakeDenoise
from ml.denoise3D_trans import MakeDenoise3DTrans
from ml.linear import MakeLinear
from ml.dense import MakeDense
from ml.dense_trans import MakeDenseTrans
from ml.ml_utils import NameModel, ParseModelName, TriangleWaveLR, NoisyDecayLR, NoisySinLR, TriangleFractalLR
from ml.custom_keras_layers import TransformerBlock, RecombineLayer
from misc.misc_utils import SavePredData
#====================================================================
def TrainKerasModel(config_path, model_name=None, model_save_path='/Users/joshuamiller/Documents/Lancaster/SavedModels', prefix=''):
    #----------------------------------------------------------------
    ''' Check for GPU access '''
    
    print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")
    print(f"TensorFlow version: {tf.__version__}")
    #----------------------------------------------------------------
    ''' Get data from config '''

    with open(config_path, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    assert (config['MODEL_TYPE'] in ['Linear', 'Dense', 'Conv', 'LSTM', 'ConvLSTM', 'RBDN', 'Split', 'Denoise', 'DenoiseTrans', 'Trans', 'ConvTrans', 'ConvLSTMTrans']), "To use this, 'MODEL_TYPE' must be 'Linear', 'Dense', 'Conv', 'ConvLSTM', 'RBDN', 'Split', 'Denoise', 'DenoiseTrans', 'Trans', 'ConvTrans', 'ConvLSTMTrans'. Got: "+str(config['MODEL_TYPE'])

    patience = config['PATIENCE']
    model_fig_save_path = config['MODEL_FIG_SAVE_PATH']
    #----------------------------------------------------------------
    ''' Get data '''
    x_data = DataLoader('config.yml', 'data_utils/data_utils_config.yml', 'HISTORY_DATA')
    y_data = DataLoader('config.yml', 'data_utils/data_utils_config.yml', 'TARGET_DATA')

    histories, targets = Histories_Targets('config.yml', x_data, y_data)

    del(x_data)
    del(y_data)

    x_train, x_test, y_train, y_test = Train_Test_Split('config.yml', histories, targets, shuffle=True)

    del(histories)
    del(targets)
    #----------------------------------------------------------------
    ''' Train model '''
    model = 69

    if (config['MODEL_TYPE'] == 'Linear'):
        model = MakeLinear(config_path, np.shape(x_train), np.shape(y_train))
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif (config['MODEL_TYPE'] == 'Dense'):
        model = MakeDense(config_path, np.shape(x_train), np.shape(y_train))
        num_epochs = config['HYPERPARAMETERS']['dense_hyperparams_dict']['epochs']
        batch_size = config['HYPERPARAMETERS']['dense_hyperparams_dict']['batch_size']
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif (config['MODEL_TYPE'] == 'Conv'):
        x_train = np.transpose(x_train, (0, 2, 3, 1, 4))
        y_train = np.transpose(y_train, (0, 2, 3, 1, 4))
        x_test = np.transpose(x_test, (0, 2, 3, 1, 4))
        y_test = np.transpose(y_test, (0, 2, 3, 1, 4))

        print("____________________________________________________________")
        print(" >> Conv transpose")
        print(" >> x_train:", np.shape(x_train))
        print(" >> y_train:", np.shape(y_train))
        print(" >> x_test :", np.shape(x_test))
        print(" >> y_test :", np.shape(y_test))
        print("____________________________________________________________")
        
        model = MakeConv(config_path, np.shape(x_train), np.shape(y_train))
        num_epochs = config['HYPERPARAMETERS']['conv_hyperparams_dict']['epochs']
        batch_size = config['HYPERPARAMETERS']['conv_hyperparams_dict']['batch_size']
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif (config['MODEL_TYPE'] == 'LSTM'):
        model = MakeLSTM(config_path, np.shape(x_train), np.shape(y_train))
        num_epochs = config['HYPERPARAMETERS']['lstm_hyperparams_dict']['epochs']
        batch_size = config['HYPERPARAMETERS']['lstm_hyperparams_dict']['batch_size']
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif (config['MODEL_TYPE'] == 'RBDN'):
        model = MakeRBDN(config_path, np.shape(x_train), np.shape(y_train))
        num_epochs = config['HYPERPARAMETERS']['rbdn_hyperparams_dict']['epochs']
        batch_size = config['HYPERPARAMETERS']['rbdn_hyperparams_dict']['batch_size']
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif (config['MODEL_TYPE'] == 'Split'):
        model = MakeSplitter(config_path, np.shape(x_train), np.shape(y_train))
        num_epochs = config['HYPERPARAMETERS']['split_hyperparams_dict']['epochs']
        batch_size = config['HYPERPARAMETERS']['split_hyperparams_dict']['batch_size']
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif (config['MODEL_TYPE'] == 'Denoise'):
        model = MakeDenoise(config_path, np.shape(x_train), np.shape(y_train))
        num_epochs = config['HYPERPARAMETERS']['denoise_hyperparams_dict']['epochs']
        batch_size = config['HYPERPARAMETERS']['denoise_hyperparams_dict']['batch_size']
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif (config['MODEL_TYPE'] == 'DenoiseTrans'):
        model = MakeDenoise3DTrans(config_path, np.shape(x_train), np.shape(y_train))
        num_epochs = config['HYPERPARAMETERS']['denoise_hyperparams_dict']['epochs']
        batch_size = config['HYPERPARAMETERS']['denoise_hyperparams_dict']['batch_size']
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif (config['MODEL_TYPE'] == 'ConvLSTM'):
        model = MakeConvLSTM(config_path, np.shape(x_train), np.shape(y_train))
        num_epochs = config['HYPERPARAMETERS']['convlstm_hyperparams_dict']['epochs']
        batch_size = config['HYPERPARAMETERS']['convlstm_hyperparams_dict']['batch_size']
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif (config['MODEL_TYPE'] == 'Trans'):
        model = MakeDenseTrans(config_path, np.shape(x_train), np.shape(y_train))
        num_epochs = config['HYPERPARAMETERS']['trans_hyperparams_dict']['epochs']
        batch_size = config['HYPERPARAMETERS']['trans_hyperparams_dict']['batch_size']
    #----------------------------------------------------------------
    model.compile(loss=keras.losses.MeanSquaredError(reduction="sum_over_batch_size", 
                                                     name="MSE"),
                  optimizer=keras.optimizers.SGD())
    
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss",
                                                      patience=patience, 
                                                      restore_best_weights=True)
    #custom_lr = TriangleWaveLR(period=5)
    #custom_lr = NoisyDecayLR(num_epochs)
    #custom_lr = NoisySinLR(num_epochs)
    custom_lr = TriangleFractalLR(num_epochs, period=20)

    print("____________________________________________________________")
    print(" >> Creating model type:", config['MODEL_TYPE'])
    print("____________________________________________________________")
    print(model.summary())
    print("____________________________________________________________")
    print(" >> x_train:", np.shape(x_train))
    print(" >> y_train:", np.shape(y_train))
    print(" >> x_test :", np.shape(x_test))
    print(" >> y_test :", np.shape(y_test))
    print("____________________________________________________________")
    print(" >> Training model type:", config['MODEL_TYPE'])
    print("____________________________________________________________")
    history = model.fit(x=x_train,
                        y=y_train,
                        validation_data=(x_test, y_test),
                        batch_size=batch_size,
                        epochs=num_epochs,
                        verbose=1,
                        callbacks=[custom_lr, early_stopping_cb])
    print(" >> Finished training model type:", config['MODEL_TYPE'])
    print("____________________________________________________________")
    #----------------------------------------------------------------
    ''' Save model '''
    if not (model_save_path==None):
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        
        if (model_name == None):
            model_name = NameModel(config_path, prefix=prefix)
    
        model_json = model.to_json()
        with open(os.path.join(model_save_path, model_name+'.json'), 'w') as json_file:
            json_file.write(model_json)

        model.save_weights(os.path.join(model_save_path, model_name+'.h5'))

        print(' >>')
        print(' >> Saving:', model_name)
        print(' >>')
        print(' >> Architecture file:', os.path.join(model_save_path, model_name+'.json'))
        print(' >> Weights file:', os.path.join(model_save_path, model_name+'.h5'))

        keras.utils.plot_model(model, show_shapes=True, show_layer_activations=True, to_file=os.path.join(model_fig_save_path, model_name+'.png'))
    
    return x_test, y_test, history
#====================================================================
def TestKerasModel(config_path, model_name, model_pred_path=None):
    #----------------------------------------------------------------
    ''' Get data from config '''

    with open(config_path, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    model_folder  = config['MODEL_SAVE_PATH']
    figure_folder = config['FIG_SAVE_PATH']

    info, param_dict = ParseModelName(model_name)

    # direction = info[0].split(' ')[0]
    # mystring.replace(" ", "_")
    #----------------------------------------------------------------
    ''' Get data '''

    x_data = DataLoader(config_path, 'data_utils/data_utils_config.yml', 'HISTORY_DATA')
    y_data = DataLoader(config_path, 'data_utils/data_utils_config.yml', 'TARGET_DATA')

    histories, targets = Histories_Targets('config.yml', x_data, y_data)

    del(x_data)
    del(y_data)

    x_train, x_test, y_train, y_test = Train_Test_Split('config.yml', histories, targets, shuffle=True)
    
    del(histories)
    del(targets)
    
    if ('Conv_' in model_name):
        x_train = np.transpose(x_train, (0, 2, 3, 1, 4))
        y_train = np.transpose(y_train, (0, 2, 3, 1, 4))
        x_test = np.transpose(x_test, (0, 2, 3, 1, 4))
        y_test = np.transpose(y_test, (0, 2, 3, 1, 4))
    
    print(" >> x_train:", np.shape(x_train))
    print(" >> y_train:", np.shape(y_train))
    print(" >> x_test :", np.shape(x_test))
    print(" >> y_test :", np.shape(y_test))
    print("____________________________________________________________")
    print(" >> Testing model type:", config['MODEL_TYPE'])
    x_test_orig_shape = np.shape(x_test)
    y_test_orig_shape = np.shape(y_test)

    raw_ozone = UnScale(y_test, 'data_utils/scale_files/ozone_standard.json').reshape(y_test_orig_shape[:-1])
    lon       = UnScale(x_test[..., -3], 'data_utils/scale_files/lon_minmax.json').reshape(x_test_orig_shape[:-1])
    lat       = UnScale(x_test[..., -2], 'data_utils/scale_files/lat_minmax.json').reshape(x_test_orig_shape[:-1])
    time      = UnScale(x_test[..., -1], 'data_utils/scale_files/time_minmax.json').reshape(x_test_orig_shape[:-1])

    #----------------------------------------------------------------
    ''' Get trained model '''

    model_architecture = ''
    model_weights = ''
    model = 69
    folders = os.listdir(model_folder)
    if ('.DS_Store' in folders):
        folders.remove('.DS_Store')

    for folder in folders:
        for root, dirs, files in os.walk(os.path.join(model_folder, folder)):
            for name in files:
                #print("root=",root, ", name=", name)
                if (model_name in name):
                    if name.endswith('.json'):
                        model_architecture = os.path.join(root, name)
                        print(" >> model: ", os.path.join(root, name))
                    elif name.endswith('.h5'):
                        model_weights = os.path.join(root, name)
                        print(" >> weights: ", os.path.join(root, name))

    with open(model_architecture, 'r') as json_file:
        loaded_model_json = json_file.read()

    # if ('Trans' in config['MODEL_TYPE']):
    #     model = keras.models.model_from_json(loaded_model_json, custom_objects={"TransformerBlock":TransformerBlock})
    # elif (config['MODEL_TYPE'] == 'Split'):
    #     model = keras.models.model_from_json(loaded_model_json, custom_objects={"RecombineLayer":RecombineLayer})
    # else:
    #     model = keras.models.model_from_json(loaded_model_json)

    model = keras.models.model_from_json(loaded_model_json, 
                                         custom_objects={"TransformerBlock":TransformerBlock, 
                                                         "RecombineLayer":RecombineLayer})
    print("____________________________________________________________")

    model.load_weights(model_weights)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss=keras.losses.MeanSquaredError(reduction="sum_over_batch_size", name="MSE"))
    #----------------------------------------------------------------
    ''' Test model '''

    y_pred = model.predict(x_test)

    y_pred = UnScale(np.squeeze(y_pred), 'data_utils/scale_files/ozone_standard.json').reshape(y_test_orig_shape[:-1])
    print("____________________________________________________________")
    print("orig y_pred=", np.shape(y_pred))
    if ('Conv_' in model_name):
        y_pred    = np.transpose(y_pred, (0, 3, 1, 2))
        raw_ozone = np.transpose(raw_ozone, (0, 3, 1, 2))
        lon       = np.transpose(lon, (0, 3, 1, 2))
        lat       = np.transpose(lat, (0, 3, 1, 2))
        time      = np.transpose(time, ((0, 3, 1, 2)))
    # raw_ozone = raw_ozone[:, 0, :, :].reshape(np.shape(raw_ozone)[0], 1, np.shape(raw_ozone)[2], np.shape(raw_ozone)[3])
    # y_pred    = y_pred[:, 0, :, :].reshape(np.shape(y_pred)[0], 1, np.shape(y_pred)[2], np.shape(y_pred)[3])
    print("____________________________________________________________")
    print("y_pred=", np.shape(y_pred), ", y_data=", np.shape(raw_ozone),
          ", lon=", np.shape(lon), ", lat=", np.shape(lat))
    print("____________________________________________________________")

    mse = np.mean(np.square(np.subtract(raw_ozone, y_pred)))
    print("MSE:", mse)
    print("____________________________________________________________")
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
                 param_dict['MODEL_TYPE_LONG']+'\n'+param_dict['REGION']+'\nMSE: '+str(round(mse, 10))+'\n'+str(param_dict['history_len'])+' days of data'+'\n'+str(param_dict['target_len'])+' days ahead', 
                 fontsize=12, fontweight='bold')
    
    #model_name = model_path.split('/')[-1]

    fig.savefig(os.path.join(figure_folder, model_name+'.pdf'), bbox_inches=None, pad_inches=0)
    
    #----------------------------------------------------------------
    ''' Save predictions '''
    SavePredData(config_path, model_name, y_pred, raw_ozone, time_axis)

    print(' >> Saved predictions:', os.path.join(model_pred_path, model_name+".npy"))
    print("____________________________________________________________")
    #----------------------------------------------------------------
    plt.show()
#x_test, y_test, history = TrainConvLSTM('config.yml')

#TestConvLSTM('config.yml', 'CONVLSTM_reg=SL_f=1_In=OFTUVXYD_Out=O_e=10')