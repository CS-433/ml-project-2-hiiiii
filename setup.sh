#!/bin/bash

WITH_PARKING=0

# Create directories
echo "Creating directories..."
mkdir -p checkpoints
mkdir -p data_history

# Download python libraries
echo "Downloading python libraries..."
pip3 install -r requirements.txt > /dev/null

# Download data
echo "Downloading data..."
rm -rf train_data.zip
rm -rf train_data_parking.zip
rm -rf test_data.zip
if [ $WITH_PARKING -eq 1 ]; then
    wget http://116.203.219.58:1234/train_data_parking.zip
else
    wget http://116.203.219.58:1234/train_data.zip
fi
wget http://116.203.219.58:1234/test_data.zip

echo "Unzipping data..."
rm -rf train_data
rm -rf train_data_parking
rm -rf test_data
if [ $WITH_PARKING -eq 1 ]; then
    unzip train_data_parking.zip > /dev/null
    mv train_data_parking train_data
else
    unzip train_data.zip > /dev/null
fi
unzip test_data.zip > /dev/null

echo "Done!"
