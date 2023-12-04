### Developer Umut Kaan Eren September 2023.
### Resources used to write these codes:
### Coursera Deep Learning Specialization Course Trigger Word Detection Assignment -Andrew Ng
### Youtube Channel for general idea of the model : https://www.youtube.com/watch?v=yv_WVwr6OkI
###                                                 https://www.youtube.com/watch?v=0fn7pj7Dutc
###                                                 https://www.youtube.com/watch?v=NITIefkRae0
### Github Open Sources
#######IMPORTS#######
import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import IPython
%matplotlib inline
import librosa
import librosa.display
import python_speech_features

sample_rate = 44100
num_mfcc = 16

def get_random_time_segment(segment_ms):
    """
    Gets a random time segment of duration segment_ms in a 3,000 ms audio clip.

    Arguments:
    segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")

    Returns:
    segment_time -- a tuple of (segment_start, segment_end) in ms
    """

    segment_start = np.random.randint(low=0,
                                      high=3000 - segment_ms)  # Make sure segment doesn't run past the 10sec background
    segment_end = segment_start + segment_ms - 1

    return (segment_start, segment_end)


def insert_audio_clip(background, audio_clip):
    """
    Insert a new audio segment over the background noise at a random time step

    Arguments:
    background -- a 3 second background audio recording.
    audio_clip -- the audio clip to be inserted/overlaid.
    Returns:
    new_background -- the updated background audio
    """

    # Get the duration of the audio clip in ms
    segment_ms = len(audio_clip)

    # Step 1: Use one of the helper functions to pick a random time segment onto which to insert
    # the new audio clip. (≈ 1 line)
    segment_time = get_random_time_segment(segment_ms)
    new_background = background.overlay(audio_clip, position=segment_time[0])
    return new_background


# Function: Create MFCC from given path
def calc_mfcc(path):
    # Load wavefile
    signal, fs = librosa.load(path, sr=sample_rate)

    # Create MFCCs from sound clip
    mfccs = python_speech_features.base.mfcc(signal,
                                             samplerate=fs,
                                             winlen=0.256,
                                             winstep=0.050,
                                             numcep=num_mfcc,
                                             nfilt=26,
                                             nfft=11290,
                                             preemph=0.0,
                                             ceplifter=0,
                                             appendEnergy=False,
                                             winfunc=np.hanning)
    return mfccs.transpose()


def create_training_example(backgrounds, audio, label):
    """
    Creates a training example with a given background, audio, and label.

    Arguments:
    background -- a 3 second background audio recording
    audio -- a list of audio segments of the background, negative and wakeword
    label -- 0 or 1 -- ground truth of the audio.

    Returns:
    mfcc_vector -- list of mfcc arrays of inserted audios on background.
    label_vector -- list of labels of each audio
    """
    mfcc_vector = []
    label_vector = [label] * len(backgrounds)
    for curr in range(len(backgrounds)):
        background_wav = AudioSegment.from_wav(backgrounds[curr])
        # do not add any wake or negative words on top of audio, just return background audio
        if audio == None:
            new_background = background_wav
            path = "/Users/umutkaaneren/WordDetectionProject/DataSelamBilge/cnn_train_3sec_nonww/"
        else:
            rand_audio_wav = AudioSegment.from_wav(audio[curr])
            new_background = insert_audio_clip(background_wav, rand_audio_wav)
            # Export new training example
            path = "/Users/umutkaaneren/WordDetectionProject/DataSelamBilge/cnn_train_3sec_ww/"
        new_path = path + "train" + "{l}".format(l=label) + "{k}".format(k=curr) + ".wav"
        new_background.export(new_path, format="wav")
        mfcc = calc_mfcc(new_path)
        mfcc_vector.append(mfcc)
    return mfcc_vector, label_vector

#Paths of background, wakeword and negative word folders
background_path = "/Users/umutkaaneren/WordDetectionProject/DataSelamBilge/background_3_sec/"
wakeword_path = "/Users/umutkaaneren/WordDetectionProject/DataSelamBilge/WW_train/"
negative_word_path = "/Users/umutkaaneren/Downloads/database/"


backgrounds = [background_path + file for file in os.listdir(background_path)]
wakewords = [wakeword_path + file for file in os.listdir(wakeword_path)]
negative_words = [negative_word_path + file for file in os.listdir(negative_word_path)]

negative_word_size = len(negative_words)
wakewords_size = len(wakewords)

#shuffling audio lists
random.shuffle(backgrounds)
random.shuffle(wakewords)
random.shuffle(negative_words)
## splitting shuffled background audios for inserting wake word, negative word or nothing.
bg_for_wake = backgrounds[:wakewords_size]
bg_for_negative = backgrounds[wakewords_size : wakewords_size + negative_word_size]
only_bg = backgrounds[wakewords_size + negative_word_size :]

### creating large data sets.
mfcc_ww , y_ww = create_training_example(bg_for_wake, wakewords, 1)
mfcc_nw , y_nw = create_training_example(bg_for_negative, negative_words, 0)
mfcc_none, y_none = create_training_example(only_bg, None, 0)

## combining all data
x_data = mfcc_ww + mfcc_nw + mfcc_none
y_data = y_ww + y_nw + y_none

## shuffling data before splitting into train, test and validation sets.
x_data_y_data = list(zip(x_data,y_data))
random.shuffle(x_data_y_data)
x_data, y_data = zip(* x_data_y_data)

### Splitting data into train, test and validation sets.
train_size = 0.8
val_size = 0.1
test_size = 0.1
data_size = len(x_data)

x_train = x_data[:int(data_size * train_size)]
y_train = y_data[:int(data_size * train_size)]
x_val = x_data[int(data_size * train_size):int(data_size * train_size + data_size * val_size)]
y_val = y_data[int(data_size * train_size):int(data_size * train_size + data_size * val_size)]
x_test =x_data[int(data_size * train_size + data_size * val_size) :]
y_test =y_data[int(data_size * train_size + data_size * val_size) :]

## Saving data for future use
feature_sets_file = 'all_labels_mfcc_sets.npz'
np.savez(feature_sets_file,
         x_train=x_train,
         y_train=y_train,
         x_val=x_val,
         y_val=y_val,
         x_test=x_test,
         y_test=y_test)