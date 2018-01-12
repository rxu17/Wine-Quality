# Wine-Quality

## Goal
The goal is to build a classification model to predict wine quality based on selected physicochemical attributes using R(ex, pH levels). The project focuses on the following three main steps:

- **Descriptive Statistics/Data Visualization**
- **Data Preprocessing/Engineering**
- **Multiple Machine Learning Classification Models**

## Software and Libraries
- car
- ggplot2
- VGAM
- class
- randomForest

## Description
The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine.  

Predictor variables (based on physicochemical tests): 
- fixed acidity 
- volatile acidity 
- citric acid 
- residual sugar 
- chlorides 
- free sulfur dioxide 
- total sulfur dioxide 
- density 
- pH 
- sulphates 
- alcohol 

Target variable (based on sensory data): 
- quality (score between 0 and 10)

For more info on the specific datasets provided, visit the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)

### Feature Selection 
Feature selection (of 12 attributes) is done in R by using the following methods:
- VIF analysis (removing highly correlated variables)
- RFE (Recursive Feature Elimination)
- Taking into account summary statistics (high P-value, low R^2)
- PCA (Primary Component Analysis)

### Machine Learning Techniques
- Logistic Regression
- KNN
- Random Forest
- More to come!

## Citation:

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.


