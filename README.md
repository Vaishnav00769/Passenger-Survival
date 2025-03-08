# Passenger Survival Prediction Model
## Overview
This project implements a machine learning model to predict passenger survival based on historical Titanic data. 
The model uses a Random Forest classifier and is trained with preprocessed and balanced data.
## Features
- Data preprocessing (handling missing values, encoding categorical variables, scaling numerical features)
- Dataset balancing using upsampling
- Random Forest classification model
- Model evaluation with accuracy, confusion matrix, classification report, and various plots
- Feature importance analysis
- ROC and Precision-Recall curves
## Installation
### Prerequisites
Ensure you have Python installed along with the following dependencies:
```sh
pip install pandas numpy matplotlib seaborn scikit-learn
```
## Usage
1. Clone this repository:
```sh
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
```
2. Run the model script:
```sh
python passenger_survival_model.py
```
3. Use the "predict_survival" method to predict survival based on passenger attributes.
## File Structure
- "passenger_survival_model.py": The main script containing the Random Forest model implementation.
- "Passanger.csv": The dataset used for training and evaluation.
- "README.md": This file.
## Model Evaluation
The model's performance is evaluated using:
- Accuracy score
- Confusion matrix
- Feature importance analysis
- ROC curve
- Precision-Recall curve
## Author
Vaishnav Agarwal
