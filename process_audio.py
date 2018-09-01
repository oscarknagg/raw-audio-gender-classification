"""
Run this file to process an audio file with a particular model.

Returns a csv file with prediction information and also a video file containing an animated prediction.
"""
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import resample
import numpy as np
from tqdm import tqdm
import soundfile as sf
import json

from config import PATH, LIBRISPEECH_SAMPLING_RATE
from models import *
from utils import whiten

import torch


print('Predicting {} GPU support'.format('with' if torch.cuda.is_available() else 'without'))


##############
# Parameters #
##############
model_path = PATH + '/models/max_pooling__n_layers=7__n_filters=64__downsampling=1__n_seconds=3.torch'
audio_path = PATH + '/data/Interview.flac'
step_seconds = 0.04
batchsize_for_prediction = 1


##############
# Load audio #
##############
audio, audio_sampling_rate = sf.read(audio_path)
audio_duration_seconds = audio.shape[0]*1./audio_sampling_rate
audio_duration_minutes = audio_duration_seconds/60.
print('Audio duration: {}s'.format(audio_duration_seconds))


##############
# Load model #
##############
model_type = model_path.split('/')[-1].split('__')[0]
model_name = model_path.split('/')[-1].split('.')[0]
model_params = {i.split('=')[0]: float(i.split('=')[1]) for i in model_name.split('__')[1:]}

# Here we assume that the model was trained on the LibriSpeech dataset
model_sampling_rate = LIBRISPEECH_SAMPLING_RATE/model_params['downsampling']
model_num_samples = int(model_params['n_seconds']*model_sampling_rate)

print('Model parameters determined from filename:')
print(json.dumps(model_params, indent=4))

if model_type == 'max_pooling':
    model = ConvNet(int(model_params['n_filters']), int(model_params['n_layers']))
elif model_type == 'dilated':
    model = DilatedNet(int(model_params['n_filters']), int(model_params['n_depth']), int(model_params['n_stacks']))
else:
    raise(ValueError, 'Model type not recognised.')

model.load_state_dict(torch.load(model_path))
model.double()
model.cuda()
model.eval()


######################
# Loop through audio #
######################
step_samples = int(step_seconds*model_sampling_rate)
step_samples_at_audio_rate = int(step_seconds*audio_sampling_rate)
print('Making predictions every {}s'.format(step_seconds))
print('This is every {} samples at the models sampling rate'.format(step_samples))
print('This is every {} samples at the input audio\'s sampling rate'.format(step_samples_at_audio_rate))

print('Looping through audio...')
default_shape = None
batch = []
pred = []
for lower in tqdm(range(0, audio.shape[0]-(int(model_params['n_seconds']*audio_sampling_rate)), step_samples_at_audio_rate)):
    x = audio[lower:lower+(int(model_params['n_seconds']*audio_sampling_rate))]

    # Don't predict on the last bit of audio where the duration isn't large enough
    if x.shape[0] != model_params['n_seconds']*audio_sampling_rate:
        break

    x = torch.from_numpy(x).reshape(1, -1)

    x = whiten(x)

    # For me the bottleneck is this scipy resample call, increasing batch size doesn't make it any faster
    x = torch.from_numpy(
        resample(x, model_num_samples, axis=1)
    ).reshape((1, 1, model_num_samples))

    y_hat = model(x).item()

    pred.append(y_hat)


###########################
# Create output dataframe #
###########################
segment_start_times_minutes = np.array(range(len(pred)))*step_seconds/60
df = pd.DataFrame(data={'minute': segment_start_times_minutes, 'p': pred})
df = df.assign(
    second=df['minute'].apply(lambda m: (m % 1)*60),
    # Time in seconds of the start of the prediction fragment
    t_start=df['minute']*60,
    # Time in seconds of the end of the prediction fragment
    t_end=df['minute']*60 + model_params['n_seconds'],
    # Time in seconds of the center of the prediction fragment
    t_center=df['minute']*60 + model_params['n_seconds']/2.
)
df.to_csv(PATH+'/data/results.csv', index=False)
