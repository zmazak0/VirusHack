import predicting
import librosa
import soundfile as sf
import sounddevice as sd
import requests
import numpy as np
import time


from keras.models import load_model

audio_path = 'audio/'
model_start = load_model('model\model40-2.04.hdf5')
classes = predicting.classes


def predict(audio, model):
    prob = model.predict(audio.reshape(1, 8000, 1))
    index = np.argmax(prob[0])
    return classes[index]


samples, sample_rate = librosa.load(audio_path + 'начать/начать_0.wav', sr=16000)
samples = librosa.resample(samples, sample_rate, 8000)

filename = 'audio/начать/начать_0.wav'
start, fs = sf.read(filename, dtype='float32')
sd.play(start, fs)
status = sd.wait()

if predict(samples, model_start) == 'начать':
    print(predict(samples, model_start))
    requests.get('http://localhost:5000/api/manage/start')
    time.sleep(2)
    requests.get('http://localhost:5000/api/manage/add')
    time.sleep(1)
    requests.get('http://localhost:5000/api/manage/add')
    time.sleep(2)
    requests.get('http://localhost:5000/api/manage/add')
    time.sleep(0.5)
    filename2 = 'audio/оплата/оплата_17.wav'
    start2, fs = sf.read(filename2, dtype='float32')
    sd.play(start2, fs)
    status = sd.wait()
    time.sleep(1)
    samples, sample_rate = librosa.load(audio_path + 'оплата/оплата_2.wav', sr=16000)
    samples = librosa.resample(samples, sample_rate, 8000)
    if predict(samples, model_start) == 'оплата':
        print(predict(samples, model_start))
        requests.get('http://localhost:5000/api/manage/navigate?pageName=pay')
        time.sleep(2)
        filename3 = 'audio/назад/назад_11.wav'
        start3, fs = sf.read(filename3, dtype='float32')
        sd.play(start3, fs)
        status = sd.wait()
        samples, sample_rate = librosa.load(audio_path + 'назад/назад_11.wav', sr=16000)
        samples = librosa.resample(samples, sample_rate, 8000)
        if predict(samples, model_start) == 'назад':
            print(predict(samples, model_start))
            requests.get('http://localhost:5000/api/manage/navigate?pageName=back')
            time.sleep(2)


