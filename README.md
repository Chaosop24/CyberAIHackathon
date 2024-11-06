# CyberAIHackathon
This project leverages machine learning and natural language processing (NLP) techniques to classify cybercrime complaints into specific categories and subcategories. It utilizes a dataset containing real-world cybercrime reports to assist law enforcement and organizations in effectively categorizing incidents for further investigation


Key Features

    Text Preprocessing: Implements robust preprocessing steps including tokenization, stemming, and stopword removal.
    TF-IDF Vectorization: Converts textual data into numerical form using the Term Frequency-Inverse Document Frequency (TF-IDF) technique.
    Class Imbalance Handling: Balances the dataset using the ADASYN technique to address the issue of underrepresented categories.
    Machine Learning Model: Trains an XGBoost classifier to predict the category of cybercrime based on the input text.
    Misclassification Analysis: Provides a heatmap visualization to understand common misclassifications.
    Visualization: Visualizes category and subcategory distributions to gain insights into the dataset.

Technologies Used

    Python
    NLP Libraries: NLTK, scikit-learn
    Machine Learning: XGBoost
    Data Balancing: ADASYN (Imbalanced-learn)
    Data Visualization: Matplotlib, Seaborn
    Model Persistence: Joblib

Project Structure

    train.csv: The dataset containing cybercrime reports.
    cgbmodel.py: Main Python script containing all preprocessing, model training, evaluation, and visualization steps.
    tfidf_pipeline_optimized.pkl: Saved TF-IDF model for transforming input text.
    xgb_category_model_optimized.pkl: Trained XGBoost model for predicting cybercrime categories.
    misclassification_heatmap.png: Heatmap visualizing common misclassifications.

Installation

    Clone the repository:

git clone https://github.com/Chaosop24/CyberAIHackathon.git

Navigate to the project directory:

cd cybercrime-classification

Install the required dependencies:

    pip install -r requirements.txt

Usage

    Ensure you have the dataset (train.csv) and (test.csv) in the project directory.
    Run the main script:

    python cgbmodel.py

    View the generated visualizations and classification reports.

Jupyter Notebook

A step-by-step Jupyter Notebook is also available for those who prefer an interactive coding environment. This notebook includes all the code, visualizations, and detailed explanations.
Results

    Accuracy: Achieved an overall accuracy of ~83% on the dataset.
    Misclassification Analysis: Identified and visualized common misclassification patterns using a heatmap.

Future Improvements

    Integrate Deep Learning: Experiment with advanced NLP models like BERT or GPT for improved classification accuracy.
    Media Analysis: Incorporate image and video analysis for multi-modal crime reporting.
    Real-Time Deployment: Build a real-time system to process and classify incoming cybercrime complaints.


Acknowledgments

    Dataset provided by National Cyber Crime Reporting Portal.
    Inspired by real-world challenges in categorizing cybercrime reports.
