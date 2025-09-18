# Email Spam Detection (Course Project)

## Overview
This project explores machine learning techniques for detecting spam emails. Using a dataset of ~6,000 emails labeled as *spam* or *ham* (non-spam), we implemented multiple classification algorithms and compared their performance. The objective was to identify which models deliver the best accuracy, precision, and scalability for real-world spam filtering.

## Key Features
- **Preprocessing & Feature Extraction**: 
  - Text cleaning (removal of noise, special characters, stopwords)  
  - Tokenization  
  - Feature extraction using Bag of Words (BoW) and TF-IDF  

- **Machine Learning Models Tested**:  
  - Support Vector Classifier (SVC)  
  - K-Nearest Neighbors (KNN)  
  - Naive Bayes (NB)  
  - Decision Tree (DT)  
  - Logistic Regression (LR)  
  - Random Forest (RF)  
  - AdaBoost, Bagging Classifier (BC), Extra Trees (ETC)  
  - Gradient Boosting Decision Tree (GBDT)  
  - XGBoost  

- **Evaluation Metrics**:  
  - Accuracy, Precision, Confusion Matrix, ROC Curve  

## Results
- **Top Performers**: Extra Trees Classifier (ETC), Naive Bayes (NB), and Support Vector Classifier (SVC) achieved **97% accuracy** and perfect **100% precision**.  
- **Most Balanced Model**: XGBoost (Accuracy: 96.8%, Precision: 96.5%) offered a strong balance between accuracy and scalability.  
- **Insights**: Simpler models like Naive Bayes performed surprisingly well, while more complex models like XGBoost provided efficiency and scalability for larger datasets.  

## Tools & Libraries
- Python  
- Pandas, Scikit-learn, NLTK  
- Matplotlib, Seaborn (for visualizations)  

## Team
This project was developed collaboratively as part of **ISTE 470.601 – Data Mining and Exploration (Fall 2024)**.  
- **Garima Singh** – Preprocessing, feature extraction, algorithm experimentation
- **Iman Akbar** – Preprocessing, algorithm experimentation, visualizations  
- **Joann Mathews** – Model evaluation, results analysis

## References
Dataset: [Kaggle Email Spam Detection Dataset](https://www.kaggle.com)  
Additional references listed in project report. 
