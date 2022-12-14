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
rm -rf train_data.zip
rm -rf test_data.zip
wget http://116.203.219.58:1234/train_data.zip
wget http://116.203.219.58:1234/test_data.zip

echo "Unzipping data..."
rm -rf train_data
rm -rf test_data
unzip train_data.zip > /dev/null
unzip test_data.zip > /dev/null

echo "Done!"
