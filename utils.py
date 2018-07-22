import soundfile as sf
import numpy as np


def write_sample(sample,dataset=None):
    if isinstance(sample,int):
        assert dataset is not None
        with open('/home/oscar/Desktop/temp.flac','w') as f:
            instance, label = dataset[sample]
            sf.write(f,instance,16000)
    elif isinstance(sample,np.ndarray):
        with open('/home/oscar/Desktop/temp.flac','w') as f:
            sf.write(f,sample,16000)
    else:
        raise ValueError


def normalise(instance,mean=0,rms=0.038021):
    # default is mean RMS of first 3 seconds of every dev sample
    instance = instance - instance.mean()
    instance_rms = np.sqrt(np.square(instance)).mean()
    return instance*(rms/instance_rms)