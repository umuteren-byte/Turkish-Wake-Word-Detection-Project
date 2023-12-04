### Developer Umut Kaan Eren September 2023.
### Resources used to write these codes:
### Coursera Deep Learning Specialization Course Trigger Word Detection Assignment -Andrew Ng
### Youtube Channel for general idea of the model : https://www.youtube.com/watch?v=yv_WVwr6OkI
###                                                 https://www.youtube.com/watch?v=0fn7pj7Dutc
###                                                 https://www.youtube.com/watch?v=NITIefkRae0
### Github Open Sources
######## IMPORTS ##########
import tensorflow as tf
import numpy as np
from pydub import AudioSegment
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import python_speech_features
from playsound import playsound

####### ALL CONSTANTS #####
num_mfcc = 16
fs = 44100
seconds = 3
filename = "prediction.wav"
chime_file = "chime.wav"
class_names = ["Wake Word NOT Detected", "Wake Word Detected"]

#calling trained model
model = tf.keras.models.load_model('cnn_model_10.keras')


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


print("Prediction Started: ")
i = 0
while True:
    print("Say Now: ")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, myrecording)

    audio, sample_rate = librosa.load(filename)
    mfcc = calc_mfcc(filename)
    prediction = model.predict(np.expand_dims(mfcc, axis=0))
    print(prediction)
    if prediction[0] > 0.99:
        playsound(chime_file)
        print(f"Wake Word Detected for ({i})")
        print("Confidence:", prediction[0])
        i += 1

    else:
        print(f"Wake Word NOT Detected")
        print("Confidence:", prediction[0])
