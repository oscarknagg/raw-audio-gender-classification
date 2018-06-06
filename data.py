import torch.utils.data
import soundfile as sf
import pandas as pd
import numpy as np
import os


sex_to_label = {'M': False, 'F': True}
label_to_sex = {False: 'M', True: 'F'}


class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, subset, length):
        print('Indexing data...')
        self.subset = subset
        self.fragment_length = length

        df = pd.read_csv('../data/LibriSpeech/SPEAKERS.TXT', skiprows=11, delimiter='|', error_bad_lines=False)
        df.columns = [col.strip().replace(';', '').lower() for col in df.columns]
        df = df.assign(
            sex=df['sex'].apply(lambda x: x.strip()),
            subset=df['subset'].apply(lambda x: x.strip()),
            name=df['name'].apply(lambda x: x.strip()),
        )

        # Get id -> sex mapping
        librispeech_id_to_sex = df[df['subset'] == subset][['id', 'sex']].to_dict()
        librispeech_id_to_sex = {k: v for k, v in zip(librispeech_id_to_sex['id'].values(), librispeech_id_to_sex['sex'].values())}

        n_files = 0
        datasetid = 0
        datasetid_to_filepath = {}
        datasetid_to_sex = {}
        for root, folders, files in os.walk('../data/LibriSpeech/{}/'.format(subset)):
            if len(files) == 0:
                continue

            librispeech_id = int(root.split('/')[-2])

            for f in files:
                # Skip non-sound files
                if not f.endswith('.flac'):
                    continue
                # Skip short files
                instance, samplerate = sf.read(os.path.join(root, f))
                if len(instance) <=  self.fragment_length:
                    continue

                datasetid_to_filepath[datasetid] = os.path.abspath(os.path.join(root, f))
                datasetid_to_sex[datasetid] = librispeech_id_to_sex[librispeech_id]
                datasetid += 1
                n_files += 1

        self.n_files = n_files

        self.datasetid_to_filepath = datasetid_to_filepath
        self.datasetid_to_sex = datasetid_to_sex

        print('Finished indexing data. {} usable files found.'.format(n_files))

    def __getitem__(self, index):
        instance, samplerate = sf.read(self.datasetid_to_filepath[index])
        # Choose a random sample of the file
        fragment_start_index = np.random.randint(0,len(instance)-self.fragment_length)
        instance = instance[fragment_start_index:fragment_start_index+self.fragment_length]
        sex = self.datasetid_to_sex[index]
        return instance, sex_to_label[sex]

    def __len__(self):
        return self.n_files
