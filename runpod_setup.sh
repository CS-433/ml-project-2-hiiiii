#!/bin/bash

# This script is used to setup the runpod environment

# Update the system
echo "Updating the system"
apt-get update
apt-get upgrade -y
apt-get autoremove -y

# Install the required packages
echo "Installing required packages"
apt-get install unzip ffmpeg libsm6 libxext6 nano -y

echo "Done"
