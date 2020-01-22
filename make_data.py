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
# filenames = list(maestro['midi_filename'])
data_path = 'training_data/MaestroV2.00/maestro-v2.0.0/'


# for note_bin:
# for speed in [0.95, 1, 1.05]:
#     train, exceeded = files2note_bin_examples(data_path, filenames, skip=1, starting_note=128, n_notes=256, speed=speed)
#     with open(f'training_data/note_bin_v1/nb_256_train{speed}shift.json', 'w') as f:
#         json.dump(train, f)


# for oore, get 601 so that we can use 600 at train time:
for speed in [0.9, 1, 1.1]:
    X = get_processed_oore2_data(data_path, filenames, skip=1, n_events=601, speed=speed)
    print('examples in X: ', len(X))
    with open(f'training_data/oore_v2/oore2_train_{speed}.json', 'w') as f:
        json.dump(X, f)


# with open(f'training_data/oore_v2/oore2_val.json', 'w') as f:
#     json.dump(X_val, f)


