import os
import librosa
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile  # for audio processing
import warnings

audio_path = 'audio/'

labels = os.listdir(audio_path)

no_of_recordings = []
for label in labels:
    waves = [f for f in os.listdir(audio_path + '/' + label) if f.endswith('.wav')]
    no_of_recordings.append(len(waves))

labels = ["вверх", "вес", "вниз", "восемь", "да", "два", "девять", 'добавить', 'количество', "назад",
          "начать", 'нет', "ноль", "один", "оплата", "отмена", "очистить", "пять", "семь", "сотрудник",
          "списать", "три", "удалить", "четыре", "шесть", "штрихкод"]

train_audio_path = 'audio'

all_wave = []
all_label = []
for label in labels:
    waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr=16000)
        samples = librosa.resample(samples, sample_rate, 8000)
        if len(samples) == 8000:
            all_wave.append(samples)
            all_label.append(label)

le = LabelEncoder()
y = le.fit_transform(all_label)
classes = list(le.classes_)

y = np_utils.to_categorical(y, num_classes=len(labels))
all_wave = np.array(all_wave).reshape(-1, 8000, 1)
K.clear_session()

