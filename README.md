# ADNI MRI Analysis

## Project Overview
This project involves processing and analyzing MRI data from the Alzheimer's Disease Neuroimaging Initiative (ADNI). The goal is to develop and train machine learning models to classify MRI images into different categories based on the presence of Alzheimer's disease.

## Project Structure

- `get_imgs.py`: Script to collect and organize MRI images from specified directories.
- `sort_imgs.py`: Script to sort images into training, validation, and test sets.
- `gen_data_and_intrplt.py`: Script for data generation and possibly interpolation of MRI data.
- `nn_adni3_sp.py`: Main script for setting up, training, and evaluating neural network models using TensorFlow.
- `test_accuracy.py`: Script to evaluate the accuracy of the trained models on a test dataset.
- `prep_data.py`: Contains utilities for preprocessing the data before it is fed into the neural network.

## Data Directories

- `ADNI_Complete_1Yr_1.5T/`: Contains raw MRI data files.
- `ADNI_Pictures/`: Processed images ready for model training and evaluation.

## Setup and Running

Ensure all dependencies are installed using:
