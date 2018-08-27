# raw-audio-gender-classification

This project contains the code to train a gender classification model that takes raw audio as inputs.

The weights of the model from the article can be found in the `models/` directory.

See my Medium article for more discussion.

## Instructions
### Requirements
Make a new virtualenv and instsall requirements from `requirements.txt`

### Run tests

Pytest...

### Data
Get training data here: http://www.openslr.org/12
- train-clean-100.tar.gz
- train-clean-360.tar.gz
- dev-clean.tar.gz

Place the unzipped training data into the `data/` folder so the file structure is as follows:
```
data/
    LibriSpeech/
        dev-clean/
        train-clean-100/
        train-clean-360/
        SPEAKERS.TXT
```

Please use the `SPEAKERS.TXT` supplied in the repo as I've made a few corrections to the one found at openslr.org.

### Training

Run `run_experiment.py` with the default parameters to train the model with the performance discussed in the article.

## Process audio

Run `process_audio.py`, specifying the model and audio file to use. The audio file must be a `.flac` file.

This script makes many predictions on different fragments of the target audio file and saves the results to
`data/results.csv`.

I used this script to produce the data for the video embedded in the Medium article.


## Notebooks

I have uploaded two notebooks with this project.

`Model_Performance_Investigation` gives a breakdown of the performance of the model over the different speakers in the
LibriSpeech dataset.

`Interview_Segmentation` is where I analysed the results of the `process_audio.py` script on an interview between Elton
John and Kirsty Wark.
