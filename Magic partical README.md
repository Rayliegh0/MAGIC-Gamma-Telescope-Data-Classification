# MAGIC Gamma Telescope Data Classification

## Overview
This project focuses on classifying gamma-ray signals from hadronic background noise using machine learning models. The dataset consists of attributes derived from Cherenkov telescope images, which are used to distinguish between gamma and hadron events.

## Dataset Information
The dataset includes 19,020 instances with 11 attributes:
- **fLength**: Major axis of the ellipse (mm)
- **fWidth**: Minor axis of the ellipse (mm) 
- **fSize**: Logarithm of the sum of pixel contents (number of photons)
- **fConc**: Ratio of sum of two highest pixels over fSize
- **fConc1**: Ratio of highest pixel over fSize
- **fAsym**: Distance from highest pixel to center, projected onto the major axis (mm)
- **fM3Long**: Third root of third moment along the major axis (mm) 
- **fM3Trans**: Third root of third moment along the minor axis (mm)
- **fAlpha**: Angle of the major axis with the vector to the origin (degrees)
- **fDist**: Distance from origin to center of the ellipse (mm)
- **Class**: Gamma (signal) or Hadron (background)

The dataset has an imbalanced distribution, with 12,332 gamma events and 6,688 hadron events.

## Data Preprocessing
### 1. **Loading and Understanding the Data**
The dataset is read into a pandas DataFrame, and column names are assigned for better readability. The class labels are converted into binary format (1 for gamma, 0 for hadron) to make them compatible with machine learning algorithms.

### 2. **Exploratory Data Analysis (EDA)**
Histograms are plotted for each feature to visualize the distribution across both classes. This helps in understanding the differences between gamma and hadron events.

### 3. **Train-Test Split**
The dataset is split into training (80%) and testing (20%) sets while maintaining the original class distribution. This ensures that the model learns effectively from both classes and is evaluated fairly.

### 4. **Handling Class Imbalance (SMOTE)**
Since gamma events significantly outnumber hadron events, the Synthetic Minority Over-sampling Technique (SMOTE) is applied to balance the classes. This prevents models from being biased toward the majority class and improves classification performance.

### 5. **Feature Scaling**
Standardization is performed using `StandardScaler` to ensure that all features contribute equally to the learning process. Without scaling, models like logistic regression and support vector machines may not perform optimally due to varying feature magnitudes.

## Model Training & Evaluation
Nine machine learning models are trained and compared:

- **Logistic Regression**: A linear model that estimates probabilities using the logistic function. It is simple, interpretable, and computationally efficient but struggles with non-linear relationships.
- **Decision Tree**: A hierarchical model that splits data based on feature thresholds. It captures non-linear relationships well but is prone to overfitting.
- **Random Forest**: An ensemble of decision trees that reduces overfitting by averaging multiple predictions. It is robust but computationally intensive.
- **Gradient Boosting**: A boosting algorithm that sequentially improves weak learners to minimize errors. It provides high accuracy but is sensitive to hyperparameters.
- **AdaBoost**: A boosting technique that assigns higher weights to misclassified instances, iteratively improving weak models. It performs well on structured data but is sensitive to noise.
- **Support Vector Machine (SVM)**: A model that finds the optimal hyperplane for classification. It works well in high-dimensional spaces but is computationally expensive for large datasets.
- **K-Nearest Neighbors (KNN)**: A non-parametric method that classifies data based on the majority vote of nearest neighbors. It is simple and effective but slow for large datasets.
- **XGBoost**: A highly optimized gradient boosting algorithm known for speed and performance. It excels in structured data but requires careful tuning.
- **Naive Bayes**: A probabilistic classifier that assumes feature independence. It is fast and works well with small datasets but performs poorly when feature dependencies exist.

### Training and Evaluation Steps:
1. Each model is trained using the resampled training data.
2. Predictions are made on the test set.
3. Performance is evaluated using accuracy and classification reports, which include precision, recall, and F1-score.
4. A bar chart is generated to visualize model accuracy for comparison.

## Model Performance
- **XGBoost and Random Forest achieved the highest accuracy (88%)**, making them the most effective classifiers for this dataset.
- **Support Vector Machine and Gradient Boosting also performed well (~86-87%)**, showing the effectiveness of ensemble methods.
- **Naive Bayes had the lowest performance (73%)**, likely due to its assumption of feature independence, which is not valid for this dataset.

## Conclusion
This study demonstrates the importance of preprocessing techniques like SMOTE and feature scaling in improving model performance. Ensemble methods such as Random Forest and XGBoost emerged as the best classifiers for distinguishing gamma-ray events from hadrons. Future improvements could involve hyperparameter tuning, feature selection, or deep learning approaches for further optimization.

---
This repository provides all code, data processing steps, and model evaluations required for reproducibility and further experimentation.

