# NLP-Classifiers-Analysis

## Description
We are using A bag of words algorithm to predict the sentiment analysis on a resturant's reviews dataset. Now after getting the processed data from a bag of words algorithm, we are applying different types of classifiers on the data and analysis the performance of the classifiers on this dataset.

## Dependencies and Libraries Used:
1. Pandas
2. Scikit-learn
3. nltk library

Please download stopwords before executing the related operation
Download it by running the command: >>nltk.download('stopwords')

## Classifier Models Used:
### 1. Logistic Regression
### 2. Naive Bayes
### 3. k-NearestNeighbors
### 4. Support Vector Machine (SVM)
### 5. Decision Tree Classification
### 6. Random Forests Classification
### 7. Gradient Boosted Decision Tree Classification

## Improvement Methods Used:
### kCrossValidation - for more accurate and random Testing of model
### GridSearch - for tuning the Hyper Parameters

# Result and Analysis:

# Naive Bayes:

### Accuracy = 67.37 %
### Standard Deviation = 0.0499
### Precision = 0.567
### Recall = 0.821
### F1 Score = 0.6707

# k-NearestsNeighbors:

#### Best Config: n = 5, p = 2 (Euclidean)

### Accuracy = 65.87 %
### Standard Deviation = 0.02354
### Precision = 0.6316
### Recall = 0.6186
### F1 Score = 0.625

# Decision Tree Classification:

### Accuracy = 75.35 %
### Standard Deviation = 0.0433546
### Precision = 0.7368
### Recall = 0.6796
### F1 Score = 0.70705

# Random Forests Classification:

#### Best Config: n = 73, criterion = 'entropy'


### Accuracy = 77.24 %
### Standard Deviation = 0.0201
### Precision = 0.7684
### Recall = 0.6822
### F1 Score = 0.72274

# Gradient Boosted Decision Tree Classification:

#### Best Config: loss = 'deviance', n = 113

### Accuracy = 77.5 %
### Standard Deviation = 0.0251
### Precision = 0.8211
### Recall = 0.67241
### F1 Score = 0.7394

# Logistic Regression

### Accuracy = 78.13 %
### Standard Deviation = 0.0205
### Precision = 0.7368
### Recall = 0.7071
### F1 Score = 0.7217

# Support Vector Machine (linear)

### Accuracy = 78.38 %
### Standard Deviation = 0.0202
### Precision = 0.7579
### Recall = 0.6923
### F1 Score = 0.7236

# Recommended Classifiers for the Model:
### 1. Support Vector Machine
### 2. Logistic Regression
### 3. Gradient Boosted Decision Tree Classification
### 4. Random Forests Classification


# Contribute to this repository:
#### Fork this repo and download the dataset (Restaurant_Reviews.tsv) and the template (Contribution Template) and insert your classifier after the Comment ("#INSERT HERE"), use the GridSearch accordingly for your classifier and rename the template to the name of the classifier and upload it and then create a pull request.
