
# Predicting Hospital Inpatient Costs Using Machine Learning

## Overview
This project aims to predict the total cost incurred by hospital inpatients based on the treatments and medications administered. It is a regression task implemented using traditional machine learning algorithms and a neural network model.

## Dataset
- **Dataset Name**: Hospital Inpatient Discharges (SPARCS De-Identified): 2012
- **Source**: [New York State Department of Health](https://health.data.ny.gov/Health/Hospital-Inpatient-Discharges-SPARCS-De%20Identified/u4ud-w55t/about_data)
- **Total Rows**: 2,544,543 (sampled 50,000 rows)
- **Features**: 34 columns with both numerical and categorical features

## Preprocessing
- Handled missing values (e.g., dropped rows with nulls in `APR Mortality Rate`)
- Removed outliers using Z-score method
- Converted ordered categorical columns (like Age Group) to numerical
- Applied One-Hot Encoding for categorical features

## Exploratory Data Analysis
- **Total Cost by Gender**
- **Total Charges by Type of Admission**
- **Age Group vs APR Severity of Illness**
- **Pair Plot across features**
- **Correlation Matrix for feature selection**

## Models Implemented

### 1. Linear Regression
- **R² Score**: 0.70

### 2. Gradient Boosting Regressor
- **Training R²**: 0.72  
- **Testing R²**: 0.71

### 3. Random Forest Regressor
- **R² Score**: 0.82 (Best among all ML models)

## Neural Network Model
- **Architecture**:
  - Input Layer
  - Hidden Layer 1: 64 units (ReLU)
  - Hidden Layer 2: 32 units (ReLU)
  - Output Layer: 1 unit
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Epochs**: 20
- **Observation**: Neural network showed consistent decrease in MAE and loss, outperforming traditional models like Linear Regression and Gradient Boosting.

## Key Results
- **Random Forest Regressor** performed best among traditional ML models
- **Neural Network** delivered improved performance and learning efficiency over time

## Future Work
- Hyperparameter tuning for neural network and boosting models
- Incorporate more complex deep learning architectures
- Use time-series features or temporal trends if available

## Author 
[**Karthik Mulugu**](https://www.linkedin.com/in/karthikmulugu/)

## References
- [SPARCS Dataset](https://health.data.ny.gov/Health/Hospital-Inpatient-Discharges-SPARCS-De%20Identified/u4ud-w55t/about_data)  
- [Gradient Boosting Guide](https://www.analyticsvidhya.com/blog/2021/09/gradient-boosting-algorithm-a-complete-guide-for-beginners/)  
- [Random Forest Theory](https://towardsdatascience.com/the-mathematics-of-decision-trees-random-forest-and-feature-importance-in-scikit-learn-and-spark-f2861df67e3)  
- [Scikit-learn Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html)
