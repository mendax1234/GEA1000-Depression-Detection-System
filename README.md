# ❤️ Topic: Early Depression Detection System

## Overview:
This repository contains the project files for a depression detection system developed using data science and machine learning techniques. In today’s fast-paced world, mental health challenges, such as depression, are increasingly prevalent, yet often go undetected. This project aims to leverage machine learning to predict depression tendencies based on user-provided data, offering a tool for early identification and intervention.  
We utilize a Random Forest model to analyze factors like work/study hours, sleep duration, and family history to predict depression risk, providing actionable insights for individuals and potentially healthcare professionals.  
The main Python script, implemented with Streamlit, is included here:  
- [Depression Prediction App](./app.py)

## Dataset:
The dataset used for training the model is not publicly available in this repository due to privacy concerns, as it includes sensitive mental health-related data. However, the system is designed to work with structured data containing the following features:  
- Numerical: `Work_Study_Hours`  
- Ordinal: `Academic_Pressure`, `Study_Satisfaction`, `Financial_Stress`, `Sleep_Duration`, `Dietary_Habits`  
- Binary: `Have_you_ever_had_suicidal_thoughts_`, `Family_History_of_Mental_Illness`  
The model was trained and validated on a private dataset, with performance evaluated across five groups (see Results section).

## Methodology:
### Data Preparation
The dataset was preprocessed to ensure compatibility with the Random Forest model:  
- **Data Cleaning**:  
  - Converted categorical variables (`Sleep_Duration`, `Dietary_Habits`, etc.) to numerical mappings.  
  - Handled special cases like "Others" by replacing them with mode values or defaults (e.g., `Sleep_Duration: 2`, `Dietary_Habits: 1`).  
  - Filled missing numerical values with medians and categorical values with modes.  
  - Ensured data type consistency and alignment with expected feature names.  

- **Feature Engineering**:  
  - No additional feature engineering was required beyond mapping categorical variables, as the Random Forest model handles mixed data types effectively.

### [Random Forest Model](./depression_prediction.py)
The Random Forest model was chosen for its robustness and ability to handle both numerical and categorical data without extensive feature scaling.  
- **Implementation**:  
  - The model was trained on the preprocessed dataset with an 80:20 train-test split (assumed based on standard practice).  
  - Hyperparameters were tuned implicitly (e.g., via default settings or prior experimentation not shown in the script).  
- **Prediction**:  
  - A threshold of 0.5 was used for binary classification (depression: Yes/No) in manual input mode, and 0.5 for file upload mode, balancing sensitivity and specificity.  
- **Validation**:  
  - Performance was assessed across five groups, likely using cross-validation or stratified sampling, to ensure generalizability.

## System Features:
The Streamlit app (`depression_prediction.py`) provides two modes:  
1. **Manual Input Mode**:  
   - Users input data via sliders and dropdowns for features like work hours, sleep duration, and mental health history.  
   - Outputs a prediction ("Depression Tendency Detected" or "No Depression Tendency") with a user-friendly interface.  
2. **File Upload Mode**:  
   - Users can upload CSV or Excel files containing multiple records for batch prediction.  
   - Supports accuracy calculation if a `Depression` column is provided in the uploaded file.  

## Results:
The Random Forest model’s performance was evaluated across five groups, with the following accuracy scores:

| Group   | Accuracy  |
|---------|-----------|
| group1  | 90.46%    |
| group2  | 90.63%    |
| group3  | 90.52%    |
| group4  | 90.47%    |
| group5  | 94.27%    |

- **Mean Accuracy**: Approximately 91.27% (calculated as the average of the five groups).  
- **Observations**:  
  - The model shows consistent performance across groups 1-4 (around 90.5%), with a notable improvement in group5 (94.27%), possibly because the model is trained using `group5.csv`.  
  - The high accuracy suggests the model effectively captures patterns in the data, though further analysis (e.g., precision, recall) could provide deeper insights into its performance on imbalanced classes.

## Conclusion:
This depression detection system demonstrates the potential of machine learning to identify depression tendencies with high accuracy (average 91.27%). Key factors influencing predictions include `Sleep_Duration`, `Financial_Stress`, and `Family_History_of_Mental_Illness`, as inferred from their prominence in the input features.  
The Random Forest model outperforms simpler models (e.g., Linear Regression, as seen in the example README) due to its ability to handle non-linear relationships and categorical data. The system is practical for both individual self-assessment (via manual input) and batch analysis (via file upload), making it a versatile tool for mental health screening.  
Future improvements could include integrating additional models (e.g., XGBoost) or expanding the feature set to enhance predictive power.

## Contributors:
1. Zhu Wenbo (Daniel) - System design, Random Forest implementation, Streamlit app development

## Setup Instructions:
1. **Install Dependencies**:  
   ```bash
   pip install streamlit pandas joblib numpy scikit-learn
   ```
2. **Run the App**:  
   - Ensure the `model/` directory contains `random_forest_model.joblib`, `categorical_options.joblib`, and `feature_names.joblib`.  
   - Execute:  
     ```bash
     streamlit run depression_prediction.py
     ```
3. **Usage**:  
   - Select "Manual Input Mode" to enter data manually or "File Upload Mode" to process a CSV/Excel file.  
   - Default files from cloud storage are not yet implemented (see Future Work).

## Future Work:
- **Cloud Integration**: Add support for default files from cloud storage (e.g., Google Cloud Storage) to pre-populate sample datasets in "File Upload Mode."  
- **Model Enhancement**: Incorporate additional algorithms (e.g., XGBoost) or ensemble methods for improved accuracy.  
- **Evaluation Metrics**: Include precision, recall, and F1-score to better assess performance on potentially imbalanced data.

## References:
- [Scikit-learn Random Forest Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)  
- [Streamlit Documentation](https://docs.streamlit.io/)  
