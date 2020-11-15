#!/bin/sh

# Remove all feature dataset
rm features/*
mkdir features
mkdir output

python3 features_extraction.py -d ../datasets/food5k/training -o features/training_features.hdf5
python3 features_extraction.py -d ../datasets/food5k/validation -o features/validation_features.hdf5 --pca models/pca.pkl --scaler models/scaler.pkl

python3 features_visualization.py --train features/training_features.hdf5 --test features/validation_features.hdf5

python3 train.py -d features/training_features.hdf5
python3 test.py -d features/validation_features.hdf5 -m models/model.pkl
