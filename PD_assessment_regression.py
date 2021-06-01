#!/usr/bin/env python
# coding: utf-8

# Yuanyuan Liu, 06/Apr./2021, Tampere.
# # Targets of this work
# 
# 0, This work focuses on comparing different types of features in dysarthric speech assessment. The features studied include eGeMAPS features extracted by OpenSMILE (egemaps), mel-filterbanks and pitch features (fbank_pitch), speech attribute scores (sas) output from an automatic speech attribute translator (ASAT) system, and MFCC features.
# The dysarthric speech assessment tasks include: 
# 
# 1, regression task for experts' ratings on speech intelligibility, voice impairment and overall severity of communication disorders of 15 controls and 35 PD speakers in PDSTU (PD SPEECH CORPUS OF TAMPERE UNIVERSITY)

# 2, regression task for UPDRS and UPDRS-speech of 50 PD speakers in the Spanish PD corpus: PC-GITA.

# 3, visualization of sas features.

# # Method of this work
# 
# 1, extract frame-level features (egemaps, fbank_pitch, sas, mfcc) and relative delta and delta-delta features, frame_length = 25 ms and frame_shift=10ms.
# 
# 2, speech segmentation with segment length of 1 s and shift of 0.2 s
# 
# 3, CNN model development for segment-level prediction with feature map of 1 s segment.
# 
# 4, segment-level prediction uses leave-one-speaker-out cross-validation
# 
# 5, speaker-level assessment (use the median value of segment-level predictions as the speaker-level prediction)
# 

# # Experiment
# ## A. Feature extraction 
# 
# - eGeMAPS features: use python package of opensmile installed in my thinkpad P51 computer. script stored in '/home/yuanyuan/Documents/MULAN-ACCENT/feature_extraction_opensmile.ipynb'
# 
# - fbank_pitch: use MULAN-ACCENT model developed with Kaldi, '/Users/sophia/kaldi/egs/MULAN-ACCENT/' in my mac computer, use copy_feats function.
# 
# - sas:
# use MULAN-ACCENT model developed with Kaldi, '/Users/sophia/kaldi/egs/MULAN-ACCENT/run_cnn.sh' in my mac computer.

# - mfcc:
# data for mfcc features were generated from the data of fbank_pitch.

# ## B. Data processing
# use data_preparation_regression.py.

# ## C. Run regression tasks.

# ### C.1 import modules.
import time
start = time.time()
import os
import sys
from glob import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes 
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import LeavePGroupsOut
from sklearn import mixture
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import tensorflow.keras.utils as utils


# ### C.2 Functions definition

# to generate data set for training, validation and test for leave-one-speaker-out cross-validation.
# modified on 2021-3-30. save less data.
def train_val_test(filepath_res, filename, val_num):
#     filename = 'speaker_features_label_dynamics.npz'
    data = np.load(os.path.join(filepath_res, filename), allow_pickle=True)
    data_speakers = data['speaker']
    data_features = data['features']
    data_labels = data['speech'] # for regression on PC-GITA and PDSTU
    # dictionary to store the indecies of x and y data.
    dict_test = {}
    dict_val = {}
    logo = LeaveOneGroupOut()
    print('get {} splits.'.format(logo.get_n_splits(groups=data_speakers)))
    lpgo = LeavePGroupsOut(n_groups=val_num)
    for idx_train_val, idx_test in logo.split(data_features, data_labels, data_speakers):
        idx_test = list(idx_test)
        idx_train_val = list(idx_train_val)
        speaker = np.unique(data_speakers[idx_test])[0]

        dict_test[speaker] = idx_test
        speakers_train_val = data_speakers[idx_train_val]
        labels_train_val = data_labels[idx_train_val]
        features_train_val = data_features[idx_train_val]
        iter_num = 0
        splits_num = lpgo.get_n_splits(groups=speakers_train_val)
        iter_num_random = np.random.randint(0, splits_num) # to choose two speakers for validation randomly.
        for idx_train, idx_val in lpgo.split(features_train_val, labels_train_val, speakers_train_val):
            if iter_num == iter_num_random:
                idx_val = list(idx_val)
                dict_val[speaker] = np.unique(speakers_train_val[idx_val])
                print('test {}; validation {}'.format(speaker, np.unique(speakers_train_val[idx_val])))
                break
            else:
                iter_num = iter_num + 1


    np.save(os.path.join(filepath_res, 'speakers_val.npy'), dict_val)
    np.save(os.path.join(filepath_res, 'idx_test.npy'), dict_test)

## Grads-cam algorithm
def get_imag_array(img, size):
    # 'img' is Numpy array of 2-dimension, 'array' is Numpy array of 3 dimensions (RGB).
    # In our case, 'img' is SAS feature segment of (51, 100)
    array = keras.preprocessing.image.img_to_array(img) 
    # add one dimension to transform the 'array' into a 'batch'.
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names):
    
    # First we create a model that maps the input image to the activations of the last convolutional layer.
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)
    
    # Second we create a model that maps from the activations of the last conv layer to the final class predictions.
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
#     print('shape of classifier_input: {}'.format(classifier_input.shape))
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)
    
    # The we compute the gradients of the top predicted class for our input image with respect to the activations of the last conv layer.
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
        
    # This is the gradients of the top predicted class with respect to the output feature map of the last conv layer.
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
#     print('shape of grads:{}'.format(grads.shape))
    # grads.shape: (1, 8, 51, 8)
#     print('grads.shape: {}'.format(grads.shape))
    
    # This is a vector where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
#     print('shape of pooled_grads:{}'.format(pooled_grads.shape))
    # pooled_grads.shape: (8,)
#     print('pooled_grads.shape: {}'.format(pooled_grads.shape))
    
    # We multiply each channel in the feature map array by 'how important this channel is' with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:,:,i] *= pooled_grads[i]
    
    # The channel-wise mean of the resulting feature map is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    
    # For visualization purpose, we will also normalize the heatmap between 0 and 1.
    if np.max(heatmap) > 0:
        heatmap = np.maximum(heatmap, 0) /np.max(heatmap)
    else:
#         print('max and min of heatmap is {} and {}.'.format(round(np.max(heatmap), 2), round(np.min(heatmap), 2)))
#         print('pooled_grads: {}'.format(pooled_grads))
        heatmap = np.maximum(heatmap, 0)
        
    
    return heatmap


# ### C.3 run experiment.


from numpy.random import seed
seed(1)
tf.random.set_seed(1)
sas_dim = 17
fbank_dim = 43
mfcc_dim = 39
egemaps_dim = 25
dynamic_width = 9
segment_length = 100 # 1s
segment_shift = 20 # 0.2s
segment_shift_warped = 100 # for warped sas features.

val_num = 4
filepath = '/scratch/rfyuli/MULAN-ACCENT/'
filepath_audio = '/home/rfyuli/audio/PDAUDIO/'
df_ratings_new = pd.read_excel(os.path.join(filepath_audio, 'Expert_Rater_Data_50speakers.xlsx'))
cnn_lossfunction = 'mse'
duration = 100 # 1s

attributes_sas_dynamics = ['fricative', 'glides', 'nasal', 'other', 'silence',
       'stop', 'voiced', 'vowel', 'coronal', 'dental', 'glottal', 'high',
       'labial', 'low', 'mid', 'palatal', 'velar', 'fricative_d1', 'glides_d1',
       'nasal_d1', 'other_d1', 'silence_d1', 'stop_d1', 'voiced_d1',
       'vowel_d1', 'coronal_d1', 'dental_d1', 'glottal_d1', 'high_d1',
       'labial_d1', 'low_d1', 'mid_d1', 'palatal_d1', 'velar_d1',
       'fricative_d2', 'glides_d2', 'nasal_d2', 'other_d2', 'silence_d2',
       'stop_d2', 'voiced_d2', 'vowel_d2', 'coronal_d2', 'dental_d2',
       'glottal_d2', 'high_d2', 'labial_d2', 'low_d2', 'mid_d2', 'palatal_d2',
       'velar_d2']

cnn_type = 'T-CONV2D_regression'
tf.keras.backend.set_floatx('float64')

# task = 'PC-GITA_read'
# task = 'PC-GITA_vowel'
# task = 'PC-GITA_spon'
task = sys.argv[1]
feat_type = sys.argv[2] # 'sas', 'mfcc', 'egemaps', 'fbank_pitch'
num_epochs = sys.argv[3] # 6
target = sys.argv[4] # PDSTU: 'overall', 'speech', 'voice' # PC-GITA: 'UPDRS', 'speech'

print(task, feat_type, num_epochs, type(num_epochs))


# parameters definition
if feat_type == 'fbank_pitch':
    filepath_res = os.path.join(filepath, task+'/cnn-fbank/' )
    filename = 'speaker_features_label.npz'
    feat_dim = fbank_dim
    dynamics = ''
    attributes = list(np.arange(0,43,1))
if feat_type == 'mfcc':
    filepath_res = os.path.join(filepath, task+'/mfcc/' )
    filename = 'speaker_features_label.npz'
    feat_dim = mfcc_dim
    dynamics = ''
    attributes = list(np.arange(0,39,1))
if feat_type == 'sas':
    filepath_res = os.path.join(filepath, task+'/res/cnn' )
    filename = 'speaker_features_label_dynamics.npz'
    feat_dim = sas_dim*3
    attributes = {"manner": ['fricative','glides','nasal','other','silence','stop','voiced','vowel'], 
      "place": ['coronal','dental','glottal','high','labial','low','mid','other','palatal','silence','velar']}
if feat_type == 'egemaps':
    filepath_res = os.path.join(filepath, task, 'opensmile/')
    filename = 'speaker_features_label_egemaps_lld_dynamics.npz'
    feat_dim = egemaps_dim*3
    attributes = list(np.arange(0,feat_dim,1))
        
    
# train, val, test set generation.
if os.path.exists(os.path.join(filepath_res, 'speakers_val.npy')) == False:
    print('to generate data for training, validation and test.')
#     filename = 'speaker_features_experts_ratings.npz'
    print(filename, val_num)
    train_val_test(filepath_res=filepath_res, filename=filename, val_num=val_num)
# CNN model structure for regression.
from numpy.random import seed
seed(1)
tf.keras.backend.clear_session() # https://keras.io/api/utils/backend_utils/
model = models.Sequential()
model.add(layers.Conv2D(128, (5, 1), activation='relu', padding='same',input_shape=(segment_length, feat_dim, 1)))
model.add(layers.MaxPooling2D((5, 1)))
model.add(layers.Conv2D(128, (5, 1), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 1)))
model.add(layers.Conv2D(128, (5, 1), activation='relu', padding='same'))
model.add(layers.AveragePooling2D(pool_size=(10, 1), strides=None, padding='valid'))
## add dense layer
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))
print('neural network: {}'.format(cnn_type))
model.summary()
last_conv_layer_name = 'conv2d_2'
classifier_layer_names = ['average_pooling2d','flatten', 'dense', 'dense_1']
## compile and train model
tf.random.set_seed(1)
model.compile(optimizer='RMSprop', loss='mse', metrics=['mae', 'mse'])
history_all = {}
dict_evaluation = {}
print('Regression on {} severity using {} neural network for task {}'.format(target, cnn_type, task))
filepath_model = os.path.join(filepath_res, cnn_type, target)
if os.path.exists(filepath_model)==False:
    os.makedirs(filepath_model)
print(filepath_model)
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=0, mode='auto')
checkPoint=keras.callbacks.ModelCheckpoint(filepath=filepath_model, monitor='val_loss',save_best_only=True)

print('model training and validation in {}'.format(filepath_model))

data = np.load(os.path.join(filepath_res, filename))
data_speakers = data['speaker']
data_features = data['features']
data_labels = data[target]
dict_test = np.load(os.path.join(filepath_res, 'idx_test.npy'), allow_pickle=True).item()
dict_val = np.load(os.path.join(filepath_res, 'speakers_val.npy'), allow_pickle=True).item()
speakers_uni = list(dict_test.keys())
df_predictions = pd.DataFrame(np.arange(len(data_speakers)*3).reshape(len(data_speakers), 3),                                      columns=['speaker', 'true_'+target,'predicted_'+target])
dict_attributes_weights = {}
total_valid_heatmaps_num = 0
total_num_segments = 0
i_start = 0 # to index the row of df_predictions.
i_end = 0 # to index the row of df_predictions.

for speaker in speakers_uni:
    print(speaker)
    idx_test = dict_test[speaker]
    x_test = data_features[idx_test]
    y_test = data_labels[idx_test]
    y_test_ori = y_test
    i_end = i_start + len(idx_test)

    speakers_val = dict_val[speaker]
    idx_val = []
    for i in speakers_val:
        idx_val = idx_val + dict_test[i]

    idx_val_test = list(set(idx_val).union(set(idx_test)))
    idx_train = list(set(np.arange(0, data_labels.shape[0], 1)).difference(set(idx_val_test)))

    x_train = data_features[idx_train]
    y_train = data_labels[idx_train]
    x_val = data_features[idx_val]
    y_val = data_labels[idx_val]

    ##
    # do feature normalization with dividing absolute maximum for train_val and test dataset separately.
    x_all = np.concatenate((x_train, x_val), axis=0)
    x_all_re = x_all.reshape(x_all.shape[0]*x_all.shape[1], x_all.shape[2])
    max_vec = np.max(np.abs(x_all_re), axis=0)
    x_train_norm = np.divide(x_train, max_vec)
    x_val_norm = np.divide(x_val, max_vec)
    x_test_re = x_test.reshape(x_test.shape[0]*x_test.shape[1], x_test.shape[2])
    max_vec_test = np.max(np.abs(x_test_re), axis=0)
    x_test_norm = np.divide(x_test, max_vec_test)

    # add one dimension of data.
    x_train_new = np.zeros(x_train.shape+(1,))
    x_train_new[:,:,:,0] = x_train_norm
    x_val_new = np.zeros(x_val.shape+(1,))
    x_val_new[:,:,:,0] = x_val_norm
    x_test_new = np.zeros(x_test.shape+(1,))
    x_test_new[:,:,:,0] = x_test_norm
    x_train_norm = x_train_new
    x_test_norm = x_test_new
    x_val_norm = x_val_new

    model_cur = model
    # verbose: 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 
    # Note that the progress bar is not particularly useful when logged to a file, so verbose=2 
    # is recommended when not running interactively (eg, in a production environment).
    history_all[speaker] = model_cur.fit(x_train_norm, y_train, epochs=int(num_epochs), shuffle=True, validation_data=(x_val_norm, y_val), callbacks=[earlyStopping, checkPoint], verbose=0)
    loss, mae, mse = model_cur.evaluate(x_test_norm, y_test, verbose=0)
    pred_ratings = model_cur.predict(x_test_norm).flatten()

    dict_evaluation[speaker] = [round(loss, 4), round(mae, 4), round(mse, 4)]
    df_predictions.iloc[i_start:i_end, 0] = speaker
    df_predictions.iloc[i_start:i_end, 1] = y_test
    df_predictions.iloc[i_start:i_end, 2] = pred_ratings
    i_start = i_end




    # grads-cam visualization.
    if feat_type == 'sas':

        # Prepare image
        segments_example = x_test_norm
        # time_neurons is the neuron number along time axis of the last conv2d layer.
        time_neurons = model_cur.get_layer(last_conv_layer_name).output.get_shape()[1]
        heatmaps_example = np.zeros((segments_example.shape[0], time_neurons, feat_dim))
        # to concatenate the resized heatmaps (100X51) considering effect of overlap.
    #     heatmaps_res = np.zeros((segments_example.shape[0], segment_length, feat_dim))
        heatmap_res_timelength = time_neurons*(int(segment_length/time_neurons))
        heatmaps_res = np.zeros((segments_example.shape[0], heatmap_res_timelength, feat_dim))
        total_num_segments += segments_example.shape[0]
        valid_heatmaps_num = 0
        for i in range(segments_example.shape[0]):
            img_array = get_imag_array(segments_example[i, :,:], size=(segment_length, feat_dim))
            heatmap = make_gradcam_heatmap(img_array, model_cur, last_conv_layer_name, classifier_layer_names)
            heatmap_new = np.zeros((1,)+heatmap.shape)
            heatmap_new[0,:,:] = heatmap 
            heatmaps_example[i, :, :] = heatmap
            heatmap_res = tf.keras.layers.UpSampling1D(size=int(segment_length/time_neurons))(heatmap_new)
            heatmaps_res[i, :, :] = heatmap_res

            # count the nonzeros heatmaps.
            if np.max(heatmap)>0:
                valid_heatmaps_num += 1

        heatmaps_mean = tf.reduce_sum(heatmaps_example, 0) / valid_heatmaps_num
        heatmaps_mean = heatmaps_mean.numpy()
        heatmaps_mean = heatmaps_mean / np.max(heatmaps_mean)
        if valid_heatmaps_num == 0:
            heatmaps_mean = np.zeros((heatmaps_mean.shape))
        with open(filepath_model+'/number_of_valid_heatmaps.txt', 'a') as file0:
            print('{}: {}={}, mae={}, {} valid heatmaps for {} segments.'.format(\
                                                                               speaker, target,\
                                                                               np.unique(y_test_ori)[0],\
                                                                               mae,\
                                                                               valid_heatmaps_num,\
                                                                               segments_example.shape[0]), file=file0)
        attributes_weight = tf.reduce_mean(heatmaps_mean[1:-1,:], 0)
        attributes_weight = attributes_weight.numpy()
        dict_attributes_weights[speaker] = np.around(attributes_weight, 3)
        total_valid_heatmaps_num += valid_heatmaps_num

        # show concatenated input feature maps and heatmaps together for the test speaker.
        # concatenation  of segments to recover the whole utterance of the test speaker.
        sas_filename = glob(os.path.join(filepath_res, speaker+'_*dynamics.xlsx'))
        df_speaker_sas_dynamics = pd.read_excel(sas_filename[0])
        x_test_con = df_speaker_sas_dynamics.values[0:segment_length+segment_shift*(x_test.shape[0]-1), 1:]
        heatmap_con = np.zeros(x_test_con.shape)
        idx_rep = []
        for i in range(x_test.shape[0]):
            heatmap_con[i*segment_shift:i*segment_shift+heatmap_res_timelength, :] += heatmaps_res[i, :, :]
            idx_rep += list(np.arange(i*segment_shift,i*segment_shift+heatmap_res_timelength,1))
        for i in range(x_test_con.shape[0]):
            heatmap_con[i, :] /= idx_rep.count(i)
        # save the heatmap_con as excel files.
        df_heatmap_con = pd.DataFrame(heatmap_con, columns=attributes_sas_dynamics)
        df_heatmap_con.to_excel(os.path.join(filepath_model, speaker+'_concatenated_heatmaps.xlsx'))
df_attributes_weights = pd.DataFrame.from_dict(dict_attributes_weights, orient='index', columns=attributes_sas_dynamics)
df_attributes_weights.to_excel(os.path.join(filepath_model,'speaker_mean_attributes_weights_gradscam.xlsx'))
df_evaluation = pd.DataFrame.from_dict(dict_evaluation, orient='index', columns=['loss', 'mae', 'mse'])
df_evaluation.to_excel(os.path.join(filepath_model,'model_evaluations.xlsx'))
df_predictions.to_excel(os.path.join(filepath_model, 'model_predictions.xlsx'))

with open(filepath_model+'/number_of_valid_heatmaps.txt', 'a') as file0:
    print('total number of nonzero heatmaps: {}'.format(total_valid_heatmaps_num), file=file0)
    print('total number of test segments: {}'.format(total_num_segments), file=file0)

# write a report.

# filepath_model = '/scratch/rfyuli/MULAN-ACCENT/spon/res/cnn/T-CONV2D/'
f = open(filepath_model+'/result_report.txt','w')
f.writelines([filepath_model,'\n'])
df_evaluation = pd.read_excel(os.path.join(filepath_model,'model_evaluations.xlsx'))
mae_mean = round(np.average(df_evaluation['mae']), 4)
mae_std = round(np.std(df_evaluation['mae']), 4)
mse_mean = round(np.average(df_evaluation['mse']), 4)
mse_std = round(np.std(df_evaluation['mse']), 4)
f.writelines(['- segment-level prediction for each speaker:\n'])
f.writelines(['- - mae mean:',str(mae_mean),'\n'])
f.writelines(['- - mae std:',str(mae_std),'\n'])
# f.writelines(['- - mse mean:',str(mse_mean),'\n'])
# f.writelines(['- - mse std:',str(mse_std),'\n'])
speakers = df_predictions['speaker']
test_labels = df_predictions['true_'+target]
test_predictions = np.around(df_predictions['predicted_'+target], 1)
rp, pp = np.around(pearsonr(test_labels, test_predictions), 6)
rs, ps = np.around(spearmanr(test_labels, test_predictions), 6)
f.writelines(['Pearson correlation (r, p):', str(rp),', ', str(pp), '\n'])
f.writelines(['Spearman correlation (r, p):', str(rs),', ', str(ps), '\n'])
speakers_uni = np.unique(speakers)
df_speaker_summary = pd.DataFrame(np.arange(len(speakers_uni)*3).reshape(len(speakers_uni), 3),\
                                      columns=['speaker', 'true_'+target,'predicted_'+target+'_median'])
for i in range(len(speakers_uni)):
    speaker = speakers_uni[i]
    idx_speaker = [j for j in range(len(speakers)) if speakers[j]==speaker]
    df_speaker_summary.iloc[i, 0] = speaker
    df_speaker_summary.iloc[i, 1] = np.unique(test_labels[idx_speaker])
    df_speaker_summary.iloc[i, 2] = np.median(test_predictions[idx_speaker])
df_speaker_summary.to_excel(filepath_model+'/speaker_summary.xlsx')
y_true = df_speaker_summary['true_'+target]
y_pred = df_speaker_summary['predicted_'+target+'_median']
mae = np.round(tf.keras.losses.MAE(y_true, y_pred), 4)
rp, pp = np.around(pearsonr(y_true, y_pred), 6)
rs, ps = np.around(spearmanr(y_true, y_pred), 6)
f.writelines(['- speaker-level prediction:\n'])
f.writelines(['- - MAE: ',str(mae), '\n'])
f.writelines(['- - Pearson correlation (r, p):', str(rp),', ', str(pp), '\n'])
f.writelines(['- - Spearman correlation (r, p):', str(rs),', ', str(ps), '\n'])

end = time.time()
running_time = round((end-start)/60, 2)
print('- running time: {} minutes'.format(running_time))
f.writelines(['- running time: ',str(running_time), ' minutes', '\n'])
f.close()