# raw-audio-gender-classification

Get training data here: http://www.openslr.org/12
- train-clean-100.tar.gz
- dev-clean.tar.gz

Please use the `SPEAKERS.txt` file in the data folder as I've fixed 
a corrupt row present in the openslr version. I've also corrected the
gender of speaker `JenniferRutters` from M to F.

TODO:
* Experiments (make notebook for each)
    1. Compare simple convolutional vs dilated convolutional models
    2. How much receptive field do we need?
    3. How much downsampling can we perform and keep performance?
    4. How many seconds of audio do we need for good classification performance?
* Commit final best model
* Record our own voices and see if the model gets it right!