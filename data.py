import torch.utils.data
import soundfile
import pandas as pd
import os


class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.subset = subset

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

            # if librispeech_id == 60:
            #     # Dodgy value
            #     continue

            for f in files:
                datasetid_to_filepath[datasetid] = os.path.abspath(os.path.join(root, f))
                datasetid_to_sex[datasetid] = librispeech_id_to_sex[librispeech_id]
                datasetid += 1

            n_files += len(files)

        self.n_files = n_files

        self.datasetid_to_filepath = datasetid_to_filepath
        self.datasetid_to_sex = datasetid_to_sex

    def __getitem__(self, index):
        return self.datasetid_to_filepath[index], self.datasetid_to_sex[index]

    def __len__(self):
        return self.n_files
