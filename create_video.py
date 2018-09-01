"""
This file reads a results file produced by process_audio.py and creates a simple video
"""
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import os

from config import PATH


##############
# Parameters #
##############
results_path = PATH + '/data/results.csv'
smoothing_window_seconds = 1
male_name = 'Elton'
female_name = 'Kirsty'


#################
# Load results  #
#################
df = pd.read_csv(results_path)
step_seconds = df['t_start'].values[1] - df['t_start'].values[0]
fragment_length_seconds = df['t_end'].values[0] - df['t_start'].values[0]
frames_per_second = int(1/step_seconds)


os.mkdir(PATH + '/plots/segmentation_video/')


##############################
# Loop through probabilities #
##############################
# Pad with NaN % (insufficient time window) at first
frame_index = 0
for _ in xrange(int(fragment_length_seconds / step_seconds)):
    img = np.zeros((168, 512, 3), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Insufficient audio', (50, 100), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(PATH + '/plots/segmentation_video/{}.png'.format(frame_index), img)
    frame_index += 1

for p_ in tqdm(df['p']):
    if p_ > 0.5:
        probability_text = '{}:{} - {} {}%'.format(
            str(int(frame_index * step_seconds / 60.)).zfill(2),
            str(int(frame_index * step_seconds) % 60).zfill(2),
            female_name,
            round(100. * (p_ - 0.5) * 2, 1)
        )
    else:
        probability_text = '{}:{}: - {} {}%'.format(
            str(int(frame_index * step_seconds / 60.)).zfill(2),
            str(int(frame_index * step_seconds) % 60).zfill(2),
            male_name,
            round(100. * (1 - 2 * p_), 1)
        )

    if np.isnan(p_):
        continue

    # Create an image with OpenCV
    img = np.zeros((168, 512, 3), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Save the image
    cv2.putText(img, probability_text.split(' - ')[-1], (50, 100), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(PATH + '/plots/segmentation_video/{}.png'.format(frame_index), img)

    frame_index += 1


##################################
# Create video by running ffmpeg #
##################################
command_template = 'ffmpeg -r {} -i {}/plots/segmentation_video/%01d.png -vcodec mpeg4 -y {}/plots/segmentation.mp4'
command = command_template.format(
    frames_per_second,
    PATH,
    PATH
)
os.system(command)


##################
# Clean up files #
##################
os.system('rm -rf {}'.format(PATH + '/plots/segmentation_video/'))
