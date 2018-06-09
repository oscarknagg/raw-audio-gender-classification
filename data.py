import torch.utils.data
import soundfile as sf
import pandas as pd
import numpy as np
import os


sex_to_label = {'M': False, 'F': True}
label_to_sex = {False: 'M', True: 'F'}


class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, subset, length, stochastic=True):
        print('Indexing data...')
        self.subset = subset
        self.fragment_length = length
        self.stochastic = stochastic

        df = pd.read_csv('../data/LibriSpeech/SPEAKERS.TXT', skiprows=11, delimiter='|', error_bad_lines=False)
        df.columns = [col.strip().replace(';', '').lower() for col in df.columns]
        df = df.assign(
            sex=df['sex'].apply(lambda x: x.strip()),
            subset=df['subset'].apply(lambda x: x.strip()),
            name=df['name'].apply(lambda x: x.strip()),
        )

        # Get id -> sex mapping
        librispeech_id_to_sex = df[df['subset'] == subset][['id', 'sex']].to_dict()
        self.librispeech_id_to_sex = {k: v for k, v in
                                 zip(librispeech_id_to_sex['id'].values(), librispeech_id_to_sex['sex'].values())}
        librispeech_id_to_name = df[df['subset'] == subset][['id', 'sex']].to_dict()
        self.librispeech_id_to_name = {k: v for k, v in
                                 zip(librispeech_id_to_sex['id'].values(), librispeech_id_to_sex['name'].values())}

        datasetid = 0
        self.n_files = 0
        self.datasetid_to_filepath = {}
        self.datasetid_to_sex = {}
        self.datasetid_to_name = {}
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

                self.datasetid_to_filepath[datasetid] = os.path.abspath(os.path.join(root, f))
                self.datasetid_to_sex[datasetid] = self.librispeech_id_to_sex[librispeech_id]
                self.datasetid_to_name[datasetid] = self.librispeech_id_to_name[librispeech_id]
                datasetid += 1
                self.n_files += 1

        print('Finished indexing data. {} usable files found.'.format(self.n_files))

    def __getitem__(self, index):
        instance, samplerate = sf.read(self.datasetid_to_filepath[index])
        # Choose a random sample of the file
        if self.stochastic:
            fragment_start_index = np.random.randint(0,len(instance)-self.fragment_length)
        else:
            fragment_start_index = 0
        instance = instance[fragment_start_index:fragment_start_index+self.fragment_length]
        sex = self.datasetid_to_sex[index]
        return instance, sex_to_label[sex]

    def __len__(self):
        return self.n_files
