#!/bin/bash

# Create directories
echo "Creating directories..."
mkdir -p checkpoints
mkdir -p saved_images
mkdir -p data_history

# Download python libraries
echo "Downloading python libraries..."
pip3 install -r requirements.txt > /dev/null

# Download data
echo "Downloading data..."
wget http://116.203.219.58:1234/train_data_x4.zip
wget http://116.203.219.58:1234/test_data.zip

echo "Unzipping data..."
rm -rf train_data_x4
rm -rf test_data
unzip train_data_x4.zip -d train_data_x4 > /dev/null
unzip test_data.zip -d test_data > /dev/null

echo "Done!"
