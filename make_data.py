import json
import pretty_midi
import numpy as np
import pandas as pd
#import my methods
from modules.midiMethods import *
from modules.dataMethods import *
import modules.models as models
import modules.mlClasses as mlClasses

maestro = pd.read_csv('training_data/maestro-v2.0.0withPeriod.csv', index_col=0)
filenames = list(maestro[maestro['split'] == 'train']['midi_filename'])
data_path = 'training_data/MaestroV2.00/maestro-v2.0.0/'

# # get 1.05 speed data
# train_105, exceeded = files2note_bin_examples(data_path, filenames, skip=1, n_notes=256, speed=1.05)
# with open('training_data/note_bin_v1/nb_256_train1.05.json', 'w') as f:
#     json.dump(train_105, f)

# get shifted data at the three speeds
for speed in [0.95, 1, 1.05]:
    train, exceeded = files2note_bin_examples(data_path, filenames, skip=1, starting_note=128, n_notes=256, speed=speed)
    with open(f'training_data/note_bin_v1/nb_256_train{speed}shift.json', 'w') as f:
        json.dump(train, f)

