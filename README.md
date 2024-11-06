# Solar Flare Classification Model Using Support Vector Machines

This project builds a classification model to predict major solar flare events using data from the Helioseismic and Magnetic Imager on NASA’s Solar Dynamics Observatory. The goal is to classify solar events as either positive (indicating a flaring region) or negative (non-flaring) based on a variety of solar activity features.

## Table of Contents

- [Context](#context)
- [Challenge](#challenge)
- [Dataset](#dataset)
- [Goals](#goals)
- [Guidelines](#guidelines)
- [Implementation](#implementation)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)

---

## Context

Solar flares are powerful bursts of electromagnetic energy from the sun that can disrupt power grids, satellite systems, and communications on Earth. To study and predict these events, NOAA operates the Geostationary Operational Environmental Satellite System (GOES), which collects various types of solar data. Monitoring and predicting solar flares 24 hours in advance could help prevent potential disasters, although predicting these rare events remains challenging due to complex solar dynamics.

## Challenge

This project focuses on building a machine learning model that can predict whether a solar event will occur within the next 24 hours. Using data collected from solar patches (HARPs), this classification model will determine if a region is likely to produce a major solar flare. The model will be implemented as a Support Vector Machine (SVM) classifier, trained and evaluated using both the 2010-2015 and 2020-2024 datasets.

## Dataset

The data is derived from NOAA and consists of solar activity features related to various HARP regions. Two datasets are used:

1. **2010-2015 Dataset**: Used as a baseline dataset.
2. **2020-2024 Dataset**: Contains the same features but includes more recent data for evaluation.

The features are categorized into four sets:

- **Main Feature Set (FS-I)**: Core solar activity measurements.
- **Time Change Features (FS-II)**: Temporal changes in FS-I properties.
- **Historical Activity Feature (FS-III)**: Summarizes historical activity in each region.
- **Max-Min Feature (FS-IV)**: Represents maximum and minimum values within a specific timeframe.

## Goals

The primary goals are to:

1. **Preprocess and Prepare Data**: Normalize features, remove missing values, and label the data.
2. **Feature Engineering**: Combine selected feature sets into a single array for input.
3. **Build Classification Models**: Train SVM models with different feature set combinations.
4. **Evaluate Performance**: Use metrics like True Skill Score (TSS) and k-fold cross-validation.
5. **Visualize Results**: Use confusion matrices and cross-validation accuracy plots to analyze performance.
6. **Assess Performance Across Datasets**: Compare results between 2010-2015 and 2020-2024 datasets.

## Guidelines

### 1. Data Preprocessing

- **Normalization**: Normalize all features.
- **Missing Values**: Remove or impute any missing values.
- **Labeling**: Assign positive and negative labels based on solar flare activity.

### 2. Feature Engineering

Combine feature sets (FS-I, FS-II, FS-III, FS-IV) into a 2-D array. Experiment with different combinations to identify the best-performing feature set.

### 3. Classification with SVM

- Implement the SVM model using `scikit-learn`.
- Train SVMs with all feature combinations and report k-fold cross-validation accuracy for each model.

### 4. Cross-Validation and Evaluation

- **k-Fold Cross-Validation**: Compute the mean and standard deviation of accuracy for each model.
- **True Skill Score (TSS)**: Evaluate performance using TSS, especially for class-imbalanced data.

### 5. Visualization

- **Cross-Validation Accuracy Plot**: Visualize the k-fold accuracy for each feature combination.
- **Confusion Matrix**: Display a confusion matrix for each model.

### 6. Dataset Comparison

Evaluate the best-performing feature set on both 2010-2015 and 2020-2024 datasets. Answer questions about performance variations between datasets in a report.

## Results

- **Best Feature Combination**: Report the feature combination with the highest TSS score.
- **Dataset Comparison**: Compare TSS scores between the datasets and explain potential reasons for differences.
- **Class Distribution Analysis**: Present the class distributions for each dataset and discuss the impact on model performance.

## Requirements

To run this project, you will need:

- **Python 3.x**
- **Libraries**:
  - `numpy` and `pandas` for data handling
  - `scikit-learn` for the SVM model and evaluation metrics
  - `matplotlib` and `seaborn` for visualization

Install dependencies with:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
