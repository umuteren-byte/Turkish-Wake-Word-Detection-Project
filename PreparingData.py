### Developer Umut Kaan Eren September 2023.
### Resources used to write these codes:
### Coursera Deep Learning Specialization Course Trigger Word Detection Assignment -Andrew Ng
### Youtube Channel for general idea of the model : https://www.youtube.com/watch?v=yv_WVwr6OkI
###                                                 https://www.youtube.com/watch?v=0fn7pj7Dutc
###                                                 https://www.youtube.com/watch?v=NITIefkRae0
### Github Open Sources
#### IMPORTS ####################
import sounddevice as sd
from scipy.io.wavfile import write


def record_audio_and_save(save_path, n_times=50):
    """
    This function will run `n_times` and everytime you press Enter you have to speak the wake word

    Parameters
    ----------
    n_times: int, default=50
        The function will run n_times default is set to 50.

    save_path: str
        Where to save the wav file which is generated in every iteration.
    """

    input("To start recording Wake Word press Enter: ")
    user_wakeWord = input("Who are you? ")
    for i in range(n_times):
        fs = 44100
        seconds = 2

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()
        write(save_path + user_wakeWord + "_" + str(i) + ".wav", fs, myrecording)
        input(f"Press to record next or two stop press ctrl + C ({i + 1}/{n_times}): ")

def record_background_sound(save_path, n_times=50):
    """
    This function will run automatically `n_times` and record your background sounds so you can make some
    keybaord typing sound and saying something gibberish.
    Note: Keep in mind that you DON'T have to say the wake word this time.

    Parameters
    ----------
    n_times: int, default=50
        The function will run n_times default is set to 50.

    save_path: str
        Where to save the wav file which is generated in every iteration.
        Note: DON'T set it to the same directory where you have saved the wake word or it will overwrite the files.
    """

    input("To start recording your background sounds press Enter: ")
    user_background = input("Whose background is this ")
    for i in range(n_times):
        fs = 44100
        seconds = 3

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()
        write(save_path + user_background + "_" + str(i) + ".wav", fs, myrecording)
        print(f"Currently on {i+1}/{n_times}")

# Step 1: Record yourself saying the Wake Word
print("Recording the Wake Word:\n")
record_audio_and_save("audio_data/", n_times=50)
print("Recording is done, Thank you!")
# Step 2: Record your background sounds (Just let it run, it will automatically record)
print("Recording the Background sounds:\n")
record_background_sound("background_sound/", n_times=150)
