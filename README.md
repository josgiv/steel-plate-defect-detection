# Steel Plate Defect Detection - Kaggle Competition

## Overview
This project involves building machine learning models to classify defects in metal plates. The dataset consists of images of metal plates with different types of defects, and the goal is to predict the type of defect present on each plate.

The project includes the following steps:
1. Data Preprocessing: Downloading and extracting the dataset, handling missing values, scaling features, dimensionality reduction, and feature selection.
2. Model Training: Training various machine learning models using both TensorFlow/Keras and scikit-learn.
3. Model Evaluation: Evaluating the performance of individual models and an ensemble of models.
4. Generating Submission: Generating a submission file with predictions for defect types in the test dataset.
5. Computer Vision: Implementing a computer vision model to predict defects from images.

## Files
- `steel-plate-defect-detection.ipynb`: Jupyter Notebook containing the entire project code.
- `submission.csv`: Submission file containing predictions for defect types in the test dataset.
- `uploaded_image.jpg`: Temporary file to store uploaded images for defect prediction.

## Dataset
The dataset consists of two main files:
- `train.csv`: Contains training data with features and defect labels.
- `test.csv`: Contains test data with features but no defect labels.

## Data Preprocessing
- Downloading and extracting the dataset.
- Handling missing values using mean imputation.
- Scaling features using StandardScaler.
- Dimensionality reduction using PCA.
- Feature selection using SelectKBest.

## Model Training
### TensorFlow/Keras Models
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine Classifier
- K-Nearest Neighbors Classifier
- Multi-Layer Perceptron Classifier

### scikit-learn Models
- Decision Tree Classifier
- Logistic Regression
- Gaussian Naive Bayes
- Support Vector Classifier
- K-Nearest Neighbors Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- AdaBoost Classifier
- Bagging Classifier
- Extra Trees Classifier
- Ridge Classifier
- Stochastic Gradient Descent Classifier
- Passive Aggressive Classifier
- Bernoulli Naive Bayes
- Multinomial Naive Bayes
- Complement Naive Bayes

## Model Evaluation
- Evaluating individual models using accuracy scores.
- Weighted ensemble of models using voting based on accuracy scores.
- Evaluation of ensemble model performance.

## Computer Vision
- Implementation of a computer vision model using MobileNetV2.
- Prediction of defect types from uploaded images.

## Usage
To run the project, follow these steps:
1. Open the `steel-plate-defect-detection.ipynb` notebook in a Jupyter environment.
2. Execute each cell sequentially to perform data preprocessing, model training, evaluation, and computer vision prediction.

## Conclusion
This project demonstrates the process of building and evaluating machine learning models for defect classification in metal plates. By combining different models and leveraging computer vision techniques, accurate predictions can be made to assist in quality control processes.
