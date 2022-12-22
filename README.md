# CS-433 Machine Learning Project 2 - Road Segmentation

## Setup

Run the script `setup.sh` to download the required packages and data. The train data with additional parking images can be obtained by change the `WITH_PARKING` variable in `setup.sh` to `1`.

## Run

Run the python file `run.py` to train the model and generate the submission file. The run should take about 2:30 hours on a _RTX 3090_ GPU and about 3:30 hours with the additional parking images.

The number of epochs and the batch size can be modified in the `constants.py` file.

## Folders and Files

- `checkpoints/` contains the checkpoints of the trained model (created with the `setup.sh` script).
- `data/` contains the utils to load and transform the data.
- `models/` contain the model used for the project.
- `run_history/` contains the history of the training and validation loss and F1-score (created with the `setup.sh` script).
- `utils/` contains different utils used for the project.
- `constants.py` contains the constants used for the project.
- `prediction.py` contains the code to predict the masks of the test images.
- `requirements.txt` contains the required python packages.
- `run.py` is the main file to run the project.
- `setup.sh` is the script to download the required packages and data.
- `train.py` contains the code to train the model.
