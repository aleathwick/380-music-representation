import json
import pretty_midi
import numpy as np
import pandas as pd
#import my methods
from modules.midiMethods import *
from modules.dataMethods import *
import modules.models as models
import modules.mlClasses as mlClasses

# maestro = pd.read_csv('training_data/maestro-v2.0.0withPeriod.csv', index_col=0)
# filenames = list(maestro[maestro['split'] == 'validation']['midi_filename'])
# # filenames = list(maestro['midi_filename'])
# data_path = 'training_data/MaestroV2.00/maestro-v2.0.0/'


#### note_bin v2 ####
# for speed in [0.90,1.1]:
# val = files2note_bin_examples(data_path, filenames, skip=1, n_notes=220, speed=1)
# with open(f'training_data/note_bin_v2/nb_220_val.json', 'w') as f:
#     json.dump(val, f)


#### oore ####
# # for oore, get 601 so that we can use 600 at train time
# # (Maybe I corrected in the function for this already!)
# for speed in [1]:
#     X = get_processed_oore2_data(data_path, filenames, skip=1, n_events=601, speed=speed)
#     print('examples in X: ', len(X))
#     with open(f'training_data/oore_v2/oore2_val.json', 'w') as f:
#         json.dump(X, f)


# with open(f'training_data/oore_v2/oore2_val.json', 'w') as f:
#     json.dump(X_val, f)

data_path = 'training_data/oore_v2/'
#### make some chroma data ####
for speed in [0.9,1,1.1]:
    for mode in ['weighted', 'lowest']:
        with open(f'training_data/oore_v2/oore2_train_{speed}.json', 'r') as f:
            examples = json.load(f)
        print(len(examples[0]))
        chroma = oore_data2chroma(np.array(examples),  mode=mode)
        with open(f'training_data/oore_v2/oore2_train_{speed}_chroma{mode}.json', 'w') as f:
            json.dump(chroma.tolist(), f)


with open(f'training_data/oore_v2/oore2_val.json', 'r') as f:
            val = json.load(f)
chroma_val = oore_data2chroma(np.array(val),  mode='weighted')
with open(f'training_data/oore_v2/oore2_val_chromaweighted.json', 'w') as f:
    json.dump(chroma_val.tolist(), f)

chroma_val = oore_data2chroma(np.array(val),  mode='lowest')
with open(f'training_data/oore_v2/oore2_val_chromalowest.json', 'w') as f:
    json.dump(chroma_val.tolist(), f)
